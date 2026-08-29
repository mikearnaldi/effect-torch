//! Elementwise operations with broadcasting.
//!
//! Binary arithmetic includes `add`, `sub`, `mul`, `div`, `maximum`,
//! `minimum`, and `pow`. Comparisons produce `u8` masks. The module also
//! implements ternary `where_`. The float unary operations are `sqrt`,
//! `exp`, `log`, `sin`, `cos`, `tanh`, `erf`, `floor`, `ceil`,
//! `round`, `abs`, `sign`, `powf`, `relu`, and `neg`. Each kernel
//! checks that its operands broadcast to the destination shape, walks the
//! output linearly, and maps each index through `broadcast_offset` into the
//! strided or broadcast operands.
//!
//! Half-precision inputs compute in `f32` and round back. `erf` uses `libm`.
//! Broadcasting follows NumPy rules. Ranks align on the right, and size-1
//! dimensions repeat without advancing the storage offset.
//!
//! Each `*_requirements` function fixes the output. The matching `*_into`
//! function writes to a caller destination without allocating, and the plain
//! wrapper composes them.

use super::tensor::{
    CpuBuffer, CpuDestination, CpuOperationRequirements, CpuTensorRequirement, Elem, Tensor,
};
use effect_torch_runtime::{broadcast_shape, DType, Layout};
use half::{bf16, f16};

fn broadcast_offset(layout: &Layout, output_shape: &[usize], output_index: usize) -> usize {
    let extra = output_shape.len() - layout.shape().len();
    let mut source = layout.offset();
    let mut remainder = output_index;
    for dimension in (0..output_shape.len()).rev() {
        let width = output_shape[dimension].max(1);
        let coordinate = remainder % width;
        remainder /= width;
        if dimension >= extra {
            let source_dimension = dimension - extra;
            if layout.shape()[source_dimension] != 1 {
                source += coordinate * layout.strides()[source_dimension];
            }
        }
    }
    source
}

fn validate_broadcast_destination<T: Elem>(
    operation: &str,
    a: &Layout,
    b: &Layout,
    destination: &CpuDestination<'_>,
) -> Result<(), String> {
    destination.validate_current::<T>(operation)?;
    let rank = a.shape().len().max(b.shape().len());
    if destination.shape().len() != rank {
        return Err(format!(
            "{operation} destination rank mismatch: expected {rank}, got {}",
            destination.shape().len()
        ));
    }
    for dimension in 0..rank {
        let a_dimension = if dimension < rank - a.shape().len() {
            1
        } else {
            a.shape()[dimension - (rank - a.shape().len())]
        };
        let b_dimension = if dimension < rank - b.shape().len() {
            1
        } else {
            b.shape()[dimension - (rank - b.shape().len())]
        };
        if a_dimension != b_dimension && a_dimension != 1 && b_dimension != 1 {
            return Err(format!(
                "{operation} cannot broadcast {:?} with {:?}",
                a.shape(),
                b.shape()
            ));
        }
        let expected = a_dimension.max(b_dimension);
        if destination.shape()[dimension] != expected {
            return Err(format!(
                "{operation} destination shape mismatch at dimension {dimension}: expected {expected}, got {}",
                destination.shape()[dimension]
            ));
        }
    }
    Ok(())
}

fn binary_broadcast_into_impl<T: Elem>(
    operation: &str,
    a: &[T],
    a_layout: &Layout,
    b: &[T],
    b_layout: &Layout,
    destination: &mut CpuDestination<'_>,
    function: impl Fn(T, T) -> T,
) -> Result<(), String> {
    validate_broadcast_destination::<T>(operation, a_layout, b_layout, destination)?;
    destination.write_current_shaped::<T, _>(operation, |shape, output| {
        for (index, value) in output.iter_mut().enumerate() {
            *value = function(
                a[broadcast_offset(a_layout, shape, index)],
                b[broadcast_offset(b_layout, shape, index)],
            );
        }
    })
}

fn binary_u8_into_impl<T: Elem>(
    operation: &str,
    a: &[T],
    a_layout: &Layout,
    b: &[T],
    b_layout: &Layout,
    destination: &mut CpuDestination<'_>,
    function: impl Fn(T, T) -> u8,
) -> Result<(), String> {
    validate_broadcast_destination::<u8>(operation, a_layout, b_layout, destination)?;
    destination.write_current_shaped::<u8, _>(operation, |shape, output| {
        for (index, value) in output.iter_mut().enumerate() {
            *value = function(
                a[broadcast_offset(a_layout, shape, index)],
                b[broadcast_offset(b_layout, shape, index)],
            );
        }
    })
}

fn unary_into_impl<T: Elem>(
    operation: &str,
    source: &[T],
    layout: &Layout,
    destination: &mut CpuDestination<'_>,
    function: impl Fn(T) -> T,
) -> Result<(), String> {
    destination.write::<T, _>(operation, layout.shape(), |output| {
        for (index, value) in output.iter_mut().enumerate() {
            *value = function(source[super::tensor::source_index(layout, index)]);
        }
    })
}

fn binary_requirements(a: &Tensor, b: &Tensor, dtype: DType) -> CpuOperationRequirements {
    assert_eq!(a.dtype(), b.dtype(), "mixed dtypes");
    CpuOperationRequirements::without_scratch(&broadcast_shape(a.shape(), b.shape()), dtype)
}

fn unary_requirements(a: &Tensor) -> CpuOperationRequirements {
    CpuOperationRequirements::without_scratch(a.shape(), a.dtype())
}

fn allocate_output(
    requirements: CpuOperationRequirements,
    write: impl FnOnce(&mut CpuDestination<'_>) -> Result<(), String>,
) -> Tensor {
    let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
    {
        let mut destination = output
            .destination()
            .expect("new CPU tensor storage must be unique");
        write(&mut destination).expect("new CPU tensor must satisfy operation requirements");
    }
    output
}

macro_rules! binary_operation {
    (
        $wrapper:ident,
        $into:ident,
        $requirements:ident,
        $output_requirements:ident,
        $scratch_requirements:ident,
        $function:expr
    ) => {
        pub fn $requirements(&self, rhs: &Tensor) -> CpuOperationRequirements {
            binary_requirements(self, rhs, self.dtype())
        }

        pub fn $output_requirements(&self, rhs: &Tensor) -> CpuTensorRequirement {
            self.$requirements(rhs).output
        }

        pub fn $scratch_requirements(&self, _rhs: &Tensor) -> &'static [CpuTensorRequirement] {
            &[]
        }

        pub fn $into(
            &self,
            rhs: &Tensor,
            destination: &mut CpuDestination<'_>,
        ) -> Result<(), String> {
            assert_eq!(self.dtype(), rhs.dtype(), "mixed dtypes");
            match (&self.buffer, &rhs.buffer) {
                (CpuBuffer::F32(a), CpuBuffer::F32(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::F64(a), CpuBuffer::F64(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::F16(a), CpuBuffer::F16(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::U8(a), CpuBuffer::U8(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::U32(a), CpuBuffer::U32(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::I64(a), CpuBuffer::I64(b)) => binary_broadcast_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                _ => unreachable!("dtype checked before dispatch"),
            }
        }

        pub fn $wrapper(&self, rhs: &Tensor) -> Tensor {
            allocate_output(self.$requirements(rhs), |destination| {
                self.$into(rhs, destination)
            })
        }
    };
}

macro_rules! comparison_operation {
    (
        $wrapper:ident,
        $into:ident,
        $requirements:ident,
        $output_requirements:ident,
        $scratch_requirements:ident,
        $function:expr
    ) => {
        pub fn $requirements(&self, rhs: &Tensor) -> CpuOperationRequirements {
            binary_requirements(self, rhs, DType::U8)
        }

        pub fn $output_requirements(&self, rhs: &Tensor) -> CpuTensorRequirement {
            self.$requirements(rhs).output
        }

        pub fn $scratch_requirements(&self, _rhs: &Tensor) -> &'static [CpuTensorRequirement] {
            &[]
        }

        pub fn $into(
            &self,
            rhs: &Tensor,
            destination: &mut CpuDestination<'_>,
        ) -> Result<(), String> {
            assert_eq!(self.dtype(), rhs.dtype(), "mixed dtypes");
            match (&self.buffer, &rhs.buffer) {
                (CpuBuffer::F32(a), CpuBuffer::F32(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::F64(a), CpuBuffer::F64(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::F16(a), CpuBuffer::F16(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::U8(a), CpuBuffer::U8(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::U32(a), CpuBuffer::U32(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                (CpuBuffer::I64(a), CpuBuffer::I64(b)) => binary_u8_into_impl(
                    stringify!($wrapper),
                    a,
                    &self.layout,
                    b,
                    &rhs.layout,
                    destination,
                    $function,
                ),
                _ => unreachable!("dtype checked before dispatch"),
            }
        }

        pub fn $wrapper(&self, rhs: &Tensor) -> Tensor {
            allocate_output(self.$requirements(rhs), |destination| {
                self.$into(rhs, destination)
            })
        }
    };
}

macro_rules! float_unary_operation {
    (
        $wrapper:ident,
        $into:ident,
        $requirements:ident,
        $output_requirements:ident,
        $scratch_requirements:ident,
        $f32_function:expr,
        $f64_function:expr
    ) => {
        pub fn $requirements(&self) -> CpuOperationRequirements {
            assert!(self.dtype().is_float(), "requires a float dtype");
            unary_requirements(self)
        }

        pub fn $output_requirements(&self) -> CpuTensorRequirement {
            self.$requirements().output
        }

        pub fn $scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
            &[]
        }

        pub fn $into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
            assert!(self.dtype().is_float(), "requires a float dtype");
            match &self.buffer {
                CpuBuffer::F32(values) => unary_into_impl(
                    stringify!($wrapper),
                    values,
                    &self.layout,
                    destination,
                    $f32_function,
                ),
                CpuBuffer::F64(values) => unary_into_impl(
                    stringify!($wrapper),
                    values,
                    &self.layout,
                    destination,
                    $f64_function,
                ),
                CpuBuffer::F16(values) => unary_into_impl(
                    stringify!($wrapper),
                    values,
                    &self.layout,
                    destination,
                    |value: f16| f16::from_f32(($f32_function)(value.to_f32())),
                ),
                CpuBuffer::BF16(values) => unary_into_impl(
                    stringify!($wrapper),
                    values,
                    &self.layout,
                    destination,
                    |value: bf16| bf16::from_f32(($f32_function)(value.to_f32())),
                ),
                _ => unreachable!("float dtype checked before dispatch"),
            }
        }

        pub fn $wrapper(&self) -> Tensor {
            allocate_output(self.$requirements(), |destination| self.$into(destination))
        }
    };
}

impl Tensor {
    binary_operation!(
        add,
        add_into,
        add_requirements,
        add_output_requirements,
        add_scratch_requirements,
        |a, b| a + b
    );
    binary_operation!(
        sub,
        sub_into,
        sub_requirements,
        sub_output_requirements,
        sub_scratch_requirements,
        |a, b| a - b
    );
    binary_operation!(
        mul,
        mul_into,
        mul_requirements,
        mul_output_requirements,
        mul_scratch_requirements,
        |a, b| a * b
    );
    binary_operation!(
        div,
        div_into,
        div_requirements,
        div_output_requirements,
        div_scratch_requirements,
        |a, b| a / b
    );
    binary_operation!(
        maximum,
        maximum_into,
        maximum_requirements,
        maximum_output_requirements,
        maximum_scratch_requirements,
        |a, b| if a >= b { a } else { b }
    );
    binary_operation!(
        minimum,
        minimum_into,
        minimum_requirements,
        minimum_output_requirements,
        minimum_scratch_requirements,
        |a, b| if a <= b { a } else { b }
    );

    pub fn pow_requirements(&self, rhs: &Tensor) -> CpuOperationRequirements {
        assert!(self.dtype().is_float(), "pow requires float dtypes");
        binary_requirements(self, rhs, self.dtype())
    }

    pub fn pow_output_requirements(&self, rhs: &Tensor) -> CpuTensorRequirement {
        self.pow_requirements(rhs).output
    }

    pub fn pow_scratch_requirements(&self, _rhs: &Tensor) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn pow_into(
        &self,
        rhs: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        assert!(self.dtype().is_float(), "pow requires float dtypes");
        assert_eq!(self.dtype(), rhs.dtype(), "mixed dtypes");
        match (&self.buffer, &rhs.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => binary_broadcast_into_impl(
                "pow",
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                |a: f32, b: f32| a.powf(b),
            ),
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => binary_broadcast_into_impl(
                "pow",
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                |a: f64, b: f64| a.powf(b),
            ),
            (CpuBuffer::F16(a), CpuBuffer::F16(b)) => binary_broadcast_into_impl(
                "pow",
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                |a: f16, b: f16| f16::from_f32(a.to_f32().powf(b.to_f32())),
            ),
            (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => binary_broadcast_into_impl(
                "pow",
                a,
                &self.layout,
                b,
                &rhs.layout,
                destination,
                |a: bf16, b: bf16| bf16::from_f32(a.to_f32().powf(b.to_f32())),
            ),
            _ => unreachable!("float dtype checked before dispatch"),
        }
    }

    pub fn pow(&self, rhs: &Tensor) -> Tensor {
        allocate_output(self.pow_requirements(rhs), |destination| {
            self.pow_into(rhs, destination)
        })
    }

    comparison_operation!(
        eq,
        eq_into,
        eq_requirements,
        eq_output_requirements,
        eq_scratch_requirements,
        |a, b| (a == b) as u8
    );
    comparison_operation!(
        gt,
        gt_into,
        gt_requirements,
        gt_output_requirements,
        gt_scratch_requirements,
        |a, b| (a > b) as u8
    );
    comparison_operation!(
        lt,
        lt_into,
        lt_requirements,
        lt_output_requirements,
        lt_scratch_requirements,
        |a, b| (a < b) as u8
    );
    comparison_operation!(
        ge,
        ge_into,
        ge_requirements,
        ge_output_requirements,
        ge_scratch_requirements,
        |a, b| (a >= b) as u8
    );
    comparison_operation!(
        le,
        le_into,
        le_requirements,
        le_output_requirements,
        le_scratch_requirements,
        |a, b| (a <= b) as u8
    );

    pub fn where_requirements(&self, cond: &Tensor, other: &Tensor) -> CpuOperationRequirements {
        assert_eq!(cond.dtype(), DType::U8, "where condition must be u8");
        assert_eq!(self.dtype(), other.dtype(), "mixed dtypes");
        let shape = broadcast_shape(&broadcast_shape(cond.shape(), self.shape()), other.shape());
        CpuOperationRequirements::without_scratch(&shape, self.dtype())
    }

    pub fn where_output_requirements(&self, cond: &Tensor, other: &Tensor) -> CpuTensorRequirement {
        self.where_requirements(cond, other).output
    }

    pub fn where_scratch_requirements(
        &self,
        _cond: &Tensor,
        _other: &Tensor,
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn where_into(
        &self,
        cond: &Tensor,
        other: &Tensor,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        if cond.dtype() != DType::U8 {
            return Err("where condition must be u8".into());
        }
        assert_eq!(self.dtype(), other.dtype(), "mixed dtypes");
        let rank = cond
            .shape()
            .len()
            .max(self.shape().len())
            .max(other.shape().len());
        if destination.shape().len() != rank {
            return Err(format!(
                "where destination rank mismatch: expected {rank}, got {}",
                destination.shape().len()
            ));
        }
        for dimension in 0..rank {
            let dimension_of = |layout: &Layout| {
                if dimension < rank - layout.shape().len() {
                    1
                } else {
                    layout.shape()[dimension - (rank - layout.shape().len())]
                }
            };
            let c = dimension_of(&cond.layout);
            let a = dimension_of(&self.layout);
            let b = dimension_of(&other.layout);
            let expected = c.max(a).max(b);
            if (c != 1 && c != expected) || (a != 1 && a != expected) || (b != 1 && b != expected) {
                return Err("where inputs cannot be broadcast together".into());
            }
            if destination.shape()[dimension] != expected {
                return Err(format!(
                    "where destination shape mismatch at dimension {dimension}: expected {expected}, got {}",
                    destination.shape()[dimension]
                ));
            }
        }
        let CpuBuffer::U8(condition) = &cond.buffer else {
            unreachable!()
        };
        macro_rules! select {
            ($a:expr, $b:expr, $type:ty) => {
                destination.write_current_shaped::<$type, _>("where", |shape, output| {
                    for (index, value) in output.iter_mut().enumerate() {
                        *value = if condition[broadcast_offset(&cond.layout, shape, index)] != 0 {
                            $a[broadcast_offset(&self.layout, shape, index)]
                        } else {
                            $b[broadcast_offset(&other.layout, shape, index)]
                        };
                    }
                })
            };
        }
        match (&self.buffer, &other.buffer) {
            (CpuBuffer::F32(a), CpuBuffer::F32(b)) => select!(a, b, f32),
            (CpuBuffer::F64(a), CpuBuffer::F64(b)) => select!(a, b, f64),
            (CpuBuffer::F16(a), CpuBuffer::F16(b)) => select!(a, b, f16),
            (CpuBuffer::BF16(a), CpuBuffer::BF16(b)) => select!(a, b, bf16),
            (CpuBuffer::U8(a), CpuBuffer::U8(b)) => select!(a, b, u8),
            (CpuBuffer::U32(a), CpuBuffer::U32(b)) => select!(a, b, u32),
            (CpuBuffer::I64(a), CpuBuffer::I64(b)) => select!(a, b, i64),
            _ => unreachable!("dtype checked before dispatch"),
        }
    }

    pub fn where_(&self, cond: &Tensor, other: &Tensor) -> Tensor {
        allocate_output(self.where_requirements(cond, other), |destination| {
            self.where_into(cond, other, destination)
        })
    }

    pub fn neg_requirements(&self) -> CpuOperationRequirements {
        assert!(self.dtype() != DType::U8, "neg on u8");
        unary_requirements(self)
    }

    pub fn neg_output_requirements(&self) -> CpuTensorRequirement {
        self.neg_requirements().output
    }

    pub fn neg_scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn neg_into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
        assert!(self.dtype() != DType::U8, "neg on u8");
        match &self.buffer {
            CpuBuffer::F32(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: f32| {
                    -value
                })
            }
            CpuBuffer::F64(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: f64| {
                    -value
                })
            }
            CpuBuffer::F16(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: f16| {
                    -value
                })
            }
            CpuBuffer::BF16(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: bf16| {
                    -value
                })
            }
            CpuBuffer::U32(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: u32| {
                    value.wrapping_neg()
                })
            }
            CpuBuffer::I64(values) => {
                unary_into_impl("neg", values, &self.layout, destination, |value: i64| {
                    -value
                })
            }
            CpuBuffer::U8(_) => unreachable!(),
        }
    }

    pub fn neg(&self) -> Tensor {
        allocate_output(self.neg_requirements(), |destination| {
            self.neg_into(destination)
        })
    }

    pub fn powf_requirements(&self, _exponent: f64) -> CpuOperationRequirements {
        assert!(self.dtype().is_float(), "requires a float dtype");
        unary_requirements(self)
    }

    pub fn powf_output_requirements(&self, exponent: f64) -> CpuTensorRequirement {
        self.powf_requirements(exponent).output
    }

    pub fn powf_scratch_requirements(&self, _exponent: f64) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn powf_into(
        &self,
        exponent: f64,
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        assert!(self.dtype().is_float(), "requires a float dtype");
        match &self.buffer {
            CpuBuffer::F32(values) => {
                unary_into_impl("powf", values, &self.layout, destination, |value: f32| {
                    value.powf(exponent as f32)
                })
            }
            CpuBuffer::F64(values) => {
                unary_into_impl("powf", values, &self.layout, destination, |value: f64| {
                    value.powf(exponent)
                })
            }
            CpuBuffer::F16(values) => {
                unary_into_impl("powf", values, &self.layout, destination, |value: f16| {
                    f16::from_f32(value.to_f32().powf(exponent as f32))
                })
            }
            CpuBuffer::BF16(values) => {
                unary_into_impl("powf", values, &self.layout, destination, |value: bf16| {
                    bf16::from_f32(value.to_f32().powf(exponent as f32))
                })
            }
            _ => unreachable!("float dtype checked before dispatch"),
        }
    }

    pub fn powf(&self, exponent: f64) -> Tensor {
        allocate_output(self.powf_requirements(exponent), |destination| {
            self.powf_into(exponent, destination)
        })
    }

    pub fn squeeze_dims_requirements(&self, dims: &[usize]) -> CpuOperationRequirements {
        let shape: Vec<usize> = self
            .shape()
            .iter()
            .enumerate()
            .filter(|(dimension, size)| !dims.contains(dimension) || **size != 1)
            .map(|(_, &size)| size)
            .collect();
        CpuOperationRequirements::without_scratch(&shape, self.dtype())
    }

    pub fn squeeze_dims_output_requirements(&self, dims: &[usize]) -> CpuTensorRequirement {
        self.squeeze_dims_requirements(dims).output
    }

    pub fn squeeze_dims_scratch_requirements(
        &self,
        _dims: &[usize],
    ) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn squeeze_dims_into(
        &self,
        dims: &[usize],
        destination: &mut CpuDestination<'_>,
    ) -> Result<(), String> {
        let mut output_dimension = 0usize;
        for (dimension, &size) in self.shape().iter().enumerate() {
            if dims.contains(&dimension) && size == 1 {
                continue;
            }
            if destination.shape().get(output_dimension) != Some(&size) {
                return Err("squeeze_dims destination shape mismatch".into());
            }
            output_dimension += 1;
        }
        if output_dimension != destination.shape().len() {
            return Err("squeeze_dims destination shape mismatch".into());
        }
        macro_rules! copy {
            ($values:expr, $type:ty) => {
                destination.write_current::<$type, _>("squeeze_dims", |output| {
                    for (index, value) in output.iter_mut().enumerate() {
                        *value = $values[super::tensor::source_index(&self.layout, index)];
                    }
                })
            };
        }
        match &self.buffer {
            CpuBuffer::F32(values) => copy!(values, f32),
            CpuBuffer::F64(values) => copy!(values, f64),
            CpuBuffer::F16(values) => copy!(values, f16),
            CpuBuffer::BF16(values) => copy!(values, bf16),
            CpuBuffer::U8(values) => copy!(values, u8),
            CpuBuffer::U32(values) => copy!(values, u32),
            CpuBuffer::I64(values) => copy!(values, i64),
        }
    }

    pub fn squeeze_dims(&self, dims: &[usize]) -> Tensor {
        let requirements = self.squeeze_dims_requirements(dims);
        if self.layout.is_contiguous()
            && self.layout.offset() == 0
            && self.buffer.len() == self.layout.numel()
        {
            return self.view(Layout::contiguous(requirements.output.shape));
        }
        allocate_output(requirements, |destination| {
            self.squeeze_dims_into(dims, destination)
        })
    }

    float_unary_operation!(
        sqrt,
        sqrt_into,
        sqrt_requirements,
        sqrt_output_requirements,
        sqrt_scratch_requirements,
        |value: f32| value.sqrt(),
        |value: f64| value.sqrt()
    );
    float_unary_operation!(
        exp,
        exp_into,
        exp_requirements,
        exp_output_requirements,
        exp_scratch_requirements,
        |value: f32| value.exp(),
        |value: f64| value.exp()
    );
    float_unary_operation!(
        log,
        log_into,
        log_requirements,
        log_output_requirements,
        log_scratch_requirements,
        |value: f32| value.ln(),
        |value: f64| value.ln()
    );
    float_unary_operation!(
        sin,
        sin_into,
        sin_requirements,
        sin_output_requirements,
        sin_scratch_requirements,
        |value: f32| value.sin(),
        |value: f64| value.sin()
    );
    float_unary_operation!(
        cos,
        cos_into,
        cos_requirements,
        cos_output_requirements,
        cos_scratch_requirements,
        |value: f32| value.cos(),
        |value: f64| value.cos()
    );
    float_unary_operation!(
        tanh,
        tanh_into,
        tanh_requirements,
        tanh_output_requirements,
        tanh_scratch_requirements,
        |value: f32| value.tanh(),
        |value: f64| value.tanh()
    );
    float_unary_operation!(
        erf,
        erf_into,
        erf_requirements,
        erf_output_requirements,
        erf_scratch_requirements,
        |value: f32| libm::erff(value),
        |value: f64| libm::erf(value)
    );
    float_unary_operation!(
        floor,
        floor_into,
        floor_requirements,
        floor_output_requirements,
        floor_scratch_requirements,
        |value: f32| value.floor(),
        |value: f64| value.floor()
    );
    float_unary_operation!(
        ceil,
        ceil_into,
        ceil_requirements,
        ceil_output_requirements,
        ceil_scratch_requirements,
        |value: f32| value.ceil(),
        |value: f64| value.ceil()
    );
    float_unary_operation!(
        round,
        round_into,
        round_requirements,
        round_output_requirements,
        round_scratch_requirements,
        |value: f32| value.round(),
        |value: f64| value.round()
    );
    float_unary_operation!(
        abs,
        abs_into,
        abs_requirements,
        abs_output_requirements,
        abs_scratch_requirements,
        |value: f32| value.abs(),
        |value: f64| value.abs()
    );
    float_unary_operation!(
        sign,
        sign_into,
        sign_requirements,
        sign_output_requirements,
        sign_scratch_requirements,
        |value: f32| if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        },
        |value: f64| if value > 0.0 {
            1.0
        } else if value < 0.0 {
            -1.0
        } else {
            0.0
        }
    );

    pub fn relu_requirements(&self) -> CpuOperationRequirements {
        unary_requirements(self)
    }

    pub fn relu_output_requirements(&self) -> CpuTensorRequirement {
        self.relu_requirements().output
    }

    pub fn relu_scratch_requirements(&self) -> &'static [CpuTensorRequirement] {
        &[]
    }

    pub fn relu_into(&self, destination: &mut CpuDestination<'_>) -> Result<(), String> {
        match &self.buffer {
            CpuBuffer::F32(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: f32| {
                    value.max(0.0)
                })
            }
            CpuBuffer::F64(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: f64| {
                    value.max(0.0)
                })
            }
            CpuBuffer::F16(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: f16| {
                    f16::from_f32(value.to_f32().max(0.0))
                })
            }
            CpuBuffer::BF16(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: bf16| {
                    bf16::from_f32(value.to_f32().max(0.0))
                })
            }
            CpuBuffer::U8(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: u8| value)
            }
            CpuBuffer::U32(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: u32| {
                    value
                })
            }
            CpuBuffer::I64(values) => {
                unary_into_impl("relu", values, &self.layout, destination, |value: i64| {
                    value.max(0)
                })
            }
        }
    }

    pub fn relu(&self) -> Tensor {
        allocate_output(self.relu_requirements(), |destination| {
            self.relu_into(destination)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::ExecutableAllocationGuard;

    fn f32_data(tensor: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(values) = &tensor.buffer else {
            panic!()
        };
        values.as_slice().to_vec()
    }

    #[test]
    fn binary_broadcast() {
        let a = Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6.], vec![2, 3]);
        let b = Tensor::from_vec(vec![10f32, 20., 30.], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(f32_data(&c), vec![11., 22., 33., 14., 25., 36.]);
    }

    #[test]
    fn binary_strided() {
        let a = Tensor::from_vec((0..6).map(|x| x as f32).collect(), vec![2, 3]);
        let at = a.view(a.layout.permute(&[1, 0]));
        let c = at.add(&at);
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(f32_data(&c), vec![0., 6., 2., 8., 4., 10.]);
    }

    #[test]
    fn comparisons_and_where() {
        let a = Tensor::from_vec(vec![1f32, 5., 3.], vec![3]);
        let b = Tensor::from_vec(vec![2f32, 4., 3.], vec![3]);
        let m = a.gt(&b);
        let CpuBuffer::U8(values) = &m.buffer else {
            panic!()
        };
        assert_eq!(values.as_slice(), &[0, 1, 0]);
        let w = a.where_(&m, &b);
        assert_eq!(f32_data(&w), vec![2., 5., 3.]);
    }

    #[test]
    fn unary_float() {
        let a = Tensor::from_vec(vec![0f32, 1.0], vec![2]);
        let e = a.exp();
        assert!((f32_data(&e)[1] - std::f32::consts::E).abs() < 1e-6);
        let h = Tensor::from_vec(vec![f16::ONE], vec![1]).exp();
        assert_eq!(h.dtype(), DType::F16);
    }

    #[test]
    fn into_paths_match_wrappers_without_allocating() {
        let a = Tensor::from_vec((0..6).map(|value| value as f32).collect(), vec![2, 3]);
        let b = Tensor::from_vec(vec![10.0f32, 20.0, 30.0], vec![3]);
        let requirements = a.add_requirements(&b);
        assert_eq!(requirements.output.shape, vec![2, 3]);
        assert_eq!(requirements.output.bytes, 24);
        assert_eq!(requirements.scratch_bytes(), 0);
        let mut output = Tensor::empty(&requirements.output.shape, requirements.output.dtype);
        {
            let _guard = ExecutableAllocationGuard::enter();
            a.add_into(&b, &mut output.destination().unwrap()).unwrap();
        }
        assert_eq!(f32_data(&output), f32_data(&a.add(&b)));
    }

    #[test]
    fn into_rejects_wrong_destination_metadata() {
        let a = Tensor::from_vec(vec![1.0f32, 2.0], vec![2]);
        let b = Tensor::from_vec(vec![3.0f32, 4.0], vec![2]);
        let mut wrong_dtype = Tensor::empty(&[2], DType::F64);
        assert!(a
            .add_into(&b, &mut wrong_dtype.destination().unwrap())
            .is_err());
        let mut wrong_shape = Tensor::empty(&[1, 2], DType::F32);
        assert!(a
            .add_into(&b, &mut wrong_shape.destination().unwrap())
            .is_err());
    }
}
