use crate::{DType, ErasedBuffer, Layout, Placement, RuntimeId};
use std::error::Error;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayoutConstraint {
    Exact(Layout),
    Contiguous,
    AnyStrided,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingLayoutPolicy {
    Require(LayoutConstraint),
    Canonicalize { target: Layout },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingAliasing {
    MayAlias,
    Disjoint,
    Exclusive,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingDecl {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
    pub layout: BindingLayoutPolicy,
    pub aliasing: BindingAliasing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    U32,
    I64,
    F32,
    F64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    U32(u32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl ScalarValue {
    pub fn scalar_type(self) -> ScalarType {
        match self {
            ScalarValue::Bool(_) => ScalarType::Bool,
            ScalarValue::U32(_) => ScalarType::U32,
            ScalarValue::I64(_) => ScalarType::I64,
            ScalarValue::F32(_) => ScalarType::F32,
            ScalarValue::F64(_) => ScalarType::F64,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScalarDecl {
    pub name: String,
    pub scalar_type: ScalarType,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeValueKind {
    U64 {
        min: u64,
        max: u64,
    },
    U32Array {
        max_len: usize,
        element_min: u32,
        element_max: u32,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeValueDecl {
    pub name: String,
    pub kind: RuntimeValueKind,
}

impl RuntimeValueDecl {
    pub fn u64(name: impl Into<String>, min: u64, max: u64) -> Self {
        Self {
            name: name.into(),
            kind: RuntimeValueKind::U64 { min, max },
        }
    }

    pub fn u32_array(name: impl Into<String>, max_len: usize) -> Self {
        Self {
            name: name.into(),
            kind: RuntimeValueKind::U32Array {
                max_len,
                element_min: u32::MIN,
                element_max: u32::MAX,
            },
        }
    }

    pub fn validate(&self, value: &RuntimeValue) -> Result<(), RuntimeValueError> {
        match (&self.kind, value) {
            (RuntimeValueKind::U64 { min, max }, RuntimeValue::U64(value)) => {
                if min > max {
                    Err(RuntimeValueError::InvalidDeclaration {
                        name: self.name.clone(),
                    })
                } else if value < min || value > max {
                    Err(RuntimeValueError::OutOfBounds {
                        name: self.name.clone(),
                        value: *value,
                        min: *min,
                        max: *max,
                    })
                } else {
                    Ok(())
                }
            }
            (
                RuntimeValueKind::U32Array {
                    max_len,
                    element_min,
                    element_max,
                },
                RuntimeValue::U32Array(values),
            ) => {
                if values.len() > *max_len {
                    return Err(RuntimeValueError::TooLong {
                        name: self.name.clone(),
                        len: values.len(),
                        max_len: *max_len,
                    });
                }
                if element_min > element_max {
                    return Err(RuntimeValueError::InvalidDeclaration {
                        name: self.name.clone(),
                    });
                }
                for (index, value) in values.iter().copied().enumerate() {
                    if value < *element_min || value > *element_max {
                        return Err(RuntimeValueError::ArrayElementOutOfBounds {
                            name: self.name.clone(),
                            index,
                            value,
                            min: *element_min,
                            max: *element_max,
                        });
                    }
                }
                Ok(())
            }
            _ => Err(RuntimeValueError::KindMismatch {
                name: self.name.clone(),
            }),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeValue {
    U64(u64),
    U32Array(Box<[u32]>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RngDecl {
    pub counter_count: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngInvocation {
    pub seed: u64,
    pub nonce: u64,
    pub counters: Box<[u64]>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InvocationSignature {
    pub scalars: Vec<ScalarDecl>,
    pub runtime_values: Vec<RuntimeValueDecl>,
    pub rng: Option<RngDecl>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputSignature {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ProgramSignature {
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub outputs: Vec<OutputSignature>,
}

#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub bindings: Vec<ErasedBuffer>,
    pub scalars: Vec<ScalarValue>,
    pub runtime_values: Vec<RuntimeValue>,
    pub rng: Option<RngInvocation>,
}

impl ProgramSignature {
    pub fn validate_invocation(
        &self,
        runtime: RuntimeId,
        invocation: &Invocation,
    ) -> Result<(), InvocationError> {
        validate_count("bindings", self.bindings.len(), invocation.bindings.len())?;
        validate_count(
            "scalars",
            self.invocation.scalars.len(),
            invocation.scalars.len(),
        )?;
        validate_count(
            "runtime values",
            self.invocation.runtime_values.len(),
            invocation.runtime_values.len(),
        )?;

        for (index, (decl, buffer)) in self.bindings.iter().zip(&invocation.bindings).enumerate() {
            if buffer.runtime_id() != runtime {
                return Err(InvocationError::InvalidOwner {
                    binding: index,
                    expected: runtime,
                    actual: buffer.runtime_id(),
                });
            }
            if buffer.dtype() != decl.dtype {
                return Err(InvocationError::DTypeMismatch {
                    binding: index,
                    expected: decl.dtype,
                    actual: buffer.dtype(),
                });
            }
            if buffer.placement() != &decl.placement {
                return Err(InvocationError::PlacementMismatch { binding: index });
            }
            if buffer.layout().shape() != decl.shape {
                return Err(InvocationError::ShapeMismatch { binding: index });
            }
            let layout_matches = match &decl.layout {
                BindingLayoutPolicy::Require(LayoutConstraint::Exact(layout)) => {
                    buffer.layout() == layout
                }
                BindingLayoutPolicy::Require(LayoutConstraint::Contiguous) => {
                    buffer.layout().is_contiguous()
                }
                BindingLayoutPolicy::Require(LayoutConstraint::AnyStrided)
                | BindingLayoutPolicy::Canonicalize { .. } => true,
            };
            if !layout_matches {
                return Err(InvocationError::LayoutMismatch { binding: index });
            }
        }

        for (index, (decl, value)) in self
            .invocation
            .scalars
            .iter()
            .zip(&invocation.scalars)
            .enumerate()
        {
            if decl.scalar_type != value.scalar_type() {
                return Err(InvocationError::ScalarTypeMismatch { scalar: index });
            }
        }

        for (index, (decl, value)) in self
            .invocation
            .runtime_values
            .iter()
            .zip(&invocation.runtime_values)
            .enumerate()
        {
            decl.validate(value)
                .map_err(|source| InvocationError::RuntimeValue {
                    runtime_value: index,
                    source,
                })?;
        }

        match (&self.invocation.rng, &invocation.rng) {
            (None, None) => {}
            (Some(decl), Some(rng)) if rng.counters.len() == decl.counter_count as usize => {}
            (Some(decl), Some(rng)) => {
                return Err(InvocationError::RngCounterCount {
                    expected: decl.counter_count as usize,
                    actual: rng.counters.len(),
                });
            }
            (expected, actual) => {
                return Err(InvocationError::RngPresence {
                    expected: expected.is_some(),
                    actual: actual.is_some(),
                });
            }
        }
        Ok(())
    }
}

fn validate_count(
    kind: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), InvocationError> {
    if expected == actual {
        Ok(())
    } else {
        Err(InvocationError::Count {
            kind,
            expected,
            actual,
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeValueError {
    InvalidDeclaration {
        name: String,
    },
    KindMismatch {
        name: String,
    },
    OutOfBounds {
        name: String,
        value: u64,
        min: u64,
        max: u64,
    },
    TooLong {
        name: String,
        len: usize,
        max_len: usize,
    },
    ArrayElementOutOfBounds {
        name: String,
        index: usize,
        value: u32,
        min: u32,
        max: u32,
    },
}

impl fmt::Display for RuntimeValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RuntimeValueError::InvalidDeclaration { name } => {
                write!(f, "runtime value {name} has invalid bounds")
            }
            RuntimeValueError::KindMismatch { name } => {
                write!(f, "runtime value {name} has the wrong kind")
            }
            RuntimeValueError::OutOfBounds {
                name,
                value,
                min,
                max,
            } => write!(f, "runtime value {name} is {value}, outside {min}..={max}"),
            RuntimeValueError::TooLong { name, len, max_len } => write!(
                f,
                "runtime value {name} has length {len}, maximum {max_len}"
            ),
            RuntimeValueError::ArrayElementOutOfBounds {
                name,
                index,
                value,
                min,
                max,
            } => write!(
                f,
                "runtime value {name}[{index}] is {value}, outside {min}..={max}"
            ),
        }
    }
}

impl Error for RuntimeValueError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationError {
    Count {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    InvalidOwner {
        binding: usize,
        expected: RuntimeId,
        actual: RuntimeId,
    },
    DTypeMismatch {
        binding: usize,
        expected: DType,
        actual: DType,
    },
    PlacementMismatch {
        binding: usize,
    },
    ShapeMismatch {
        binding: usize,
    },
    LayoutMismatch {
        binding: usize,
    },
    ScalarTypeMismatch {
        scalar: usize,
    },
    RuntimeValue {
        runtime_value: usize,
        source: RuntimeValueError,
    },
    RngPresence {
        expected: bool,
        actual: bool,
    },
    RngCounterCount {
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for InvocationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InvocationError::Count {
                kind,
                expected,
                actual,
            } => write!(f, "expected {expected} {kind}, received {actual}"),
            InvocationError::InvalidOwner {
                binding,
                expected,
                actual,
            } => write!(
                f,
                "binding {binding} is owned by runtime {actual}, expected runtime {expected}"
            ),
            InvocationError::DTypeMismatch {
                binding,
                expected,
                actual,
            } => write!(
                f,
                "binding {binding} has dtype {actual}, expected {expected}"
            ),
            InvocationError::PlacementMismatch { binding } => {
                write!(f, "binding {binding} has the wrong placement")
            }
            InvocationError::ShapeMismatch { binding } => {
                write!(f, "binding {binding} has the wrong shape")
            }
            InvocationError::LayoutMismatch { binding } => {
                write!(f, "binding {binding} has the wrong layout")
            }
            InvocationError::ScalarTypeMismatch { scalar } => {
                write!(f, "scalar {scalar} has the wrong type")
            }
            InvocationError::RuntimeValue {
                runtime_value,
                source,
            } => write!(f, "runtime value {runtime_value} is invalid: {source}"),
            InvocationError::RngPresence { expected, actual } => write!(
                f,
                "RNG presence is {actual}, expected RNG presence to be {expected}"
            ),
            InvocationError::RngCounterCount { expected, actual } => {
                write!(f, "expected {expected} RNG counters, received {actual}")
            }
        }
    }
}

impl Error for InvocationError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            InvocationError::RuntimeValue { source, .. } => Some(source),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Buffer, DeviceId};
    use std::any::Any;

    #[derive(Debug)]
    struct TestBuffer {
        owner: RuntimeId,
        placement: Placement,
        dtype: DType,
        layout: Layout,
    }

    impl Buffer for TestBuffer {
        fn runtime_id(&self) -> RuntimeId {
            self.owner
        }

        fn placement(&self) -> &Placement {
            &self.placement
        }

        fn dtype(&self) -> DType {
            self.dtype
        }

        fn layout(&self) -> &Layout {
            &self.layout
        }

        fn as_any(&self) -> &dyn Any {
            self
        }
    }

    fn signature() -> ProgramSignature {
        ProgramSignature {
            bindings: vec![BindingDecl {
                shape: vec![2],
                dtype: DType::F32,
                placement: Placement::new(DeviceId::new("cpu:0")),
                layout: BindingLayoutPolicy::Require(LayoutConstraint::Contiguous),
                aliasing: BindingAliasing::MayAlias,
            }],
            invocation: InvocationSignature {
                runtime_values: vec![RuntimeValueDecl::u64("active", 1, 4)],
                ..InvocationSignature::default()
            },
            outputs: Vec::new(),
        }
    }

    fn invocation(owner: RuntimeId, active: u64) -> Invocation {
        Invocation {
            bindings: vec![ErasedBuffer::new(TestBuffer {
                owner,
                placement: Placement::new(DeviceId::new("cpu:0")),
                dtype: DType::F32,
                layout: Layout::contiguous(vec![2]),
            })],
            runtime_values: vec![RuntimeValue::U64(active)],
            ..Invocation::default()
        }
    }

    #[test]
    fn invocation_rejects_foreign_binding_owners() {
        let expected = RuntimeId::new();
        let foreign = RuntimeId::new();
        assert_eq!(
            signature().validate_invocation(expected, &invocation(foreign, 2)),
            Err(InvocationError::InvalidOwner {
                binding: 0,
                expected,
                actual: foreign,
            })
        );
        assert!(signature()
            .validate_invocation(expected, &invocation(expected, 2))
            .is_ok());
    }

    #[test]
    fn bounded_runtime_values_are_enforced() {
        let decl = RuntimeValueDecl::u64("active", 1, 4);
        assert!(decl.validate(&RuntimeValue::U64(1)).is_ok());
        assert!(decl.validate(&RuntimeValue::U64(4)).is_ok());
        assert_eq!(
            decl.validate(&RuntimeValue::U64(5)),
            Err(RuntimeValueError::OutOfBounds {
                name: "active".into(),
                value: 5,
                min: 1,
                max: 4,
            })
        );

        let array = RuntimeValueDecl {
            name: "blocks".into(),
            kind: RuntimeValueKind::U32Array {
                max_len: 2,
                element_min: 1,
                element_max: 8,
            },
        };
        assert!(array
            .validate(&RuntimeValue::U32Array(vec![1, 8].into_boxed_slice()))
            .is_ok());
        assert!(matches!(
            array.validate(&RuntimeValue::U32Array(vec![1, 2, 3].into_boxed_slice())),
            Err(RuntimeValueError::TooLong { .. })
        ));
        assert!(matches!(
            array.validate(&RuntimeValue::U32Array(vec![0].into_boxed_slice())),
            Err(RuntimeValueError::ArrayElementOutOfBounds { index: 0, .. })
        ));
    }
}
