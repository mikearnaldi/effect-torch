//! Program signatures and invocation validation.
//!
//! [`ProgramSignature`] defines a compiled program's calling convention. It
//! lists tensor [`BindingDecl`]s, scalar and [`RuntimeValueDecl`] parameters,
//! an optional [`RngDecl`], and [`OutputSignature`]s. Before execution, the
//! runtime checks each [`Invocation`] with
//! [`ProgramSignature::validate_invocation`].
//!
//! # Validation rules
//!
//! [`validate_invocation`](ProgramSignature::validate_invocation) checks these
//! conditions in order: argument counts, RNG presence and counter count,
//! buffer ownership, binding metadata, scalar types, and runtime-value bounds.
//! Each binding must carry the calling runtime's [`RuntimeId`]. Its dtype,
//! placement, shape, and layout must follow the declaration. Runtime values
//! must stay within declared `min..=max` ranges, array length caps, and
//! element ranges. Each failure has its own [`InvocationError`] variant.
//!
//! # Layout policies
//!
//! Each binding has a [`BindingLayoutPolicy`]. `Require(`
//! [`LayoutConstraint`]`)` requires `Exact`, `Contiguous`,
//! `ZeroOffsetContiguous`, or `AnyStrided` layout. `Canonicalize { target }`
//! accepts any valid dense layout because the backend converts it to `target`
//! before use. Packed representations retain their format-specific canonical
//! layout requirement under every binding policy.
//! [`BindingAliasing`] records whether a binding may share storage, is
//! disjoint from other bindings, or is the only writer to its storage.
//!
//! # RNG
//!
//! Randomness is explicit and replayable. [`RngDecl`] fixes the counter count
//! at compile time. Each [`RngInvocation`] supplies a seed, nonce, and exactly
//! that many counters. This prevents a frozen graph from replaying stale random
//! state.

use crate::{
    DType, ErasedBuffer, Layout, LayoutConstraintSpec, Placement, RuntimeId, StorageMetadata,
    StorageRepresentation, ValueSpec,
};
use std::error::Error;
use std::fmt;

/// Layout requirement for a binding.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayoutConstraint {
    /// The layout must equal this exact shape/strides/offset triple.
    Exact(Layout),
    /// Densely packed row-major at any offset.
    Contiguous,
    /// Densely packed row-major with offset 0.
    ZeroOffsetContiguous,
    /// Any valid dense strides are acceptable. Packed formats retain their
    /// intrinsic layout requirements.
    AnyStrided,
}

/// Layout handling at the call boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingLayoutPolicy {
    /// The caller's buffer must already satisfy the constraint.
    Require(LayoutConstraint),
    /// Accepts any valid dense layout. The backend canonicalizes it to `target`.
    /// Packed inputs and targets must satisfy their intrinsic layout requirements.
    Canonicalize { target: Layout },
}

/// Planner/backend assertion about overlap between a binding and other
/// storage.
///
/// [`ProgramSignature::validate_invocation`] records these relationships but
/// cannot prove them. It receives independent buffer handles without a
/// portable byte-range alias model. Backends that rely on `Disjoint` or
/// `Exclusive` must enforce the assertion at their ownership boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingAliasing {
    /// May share storage with other bindings.
    MayAlias,
    /// Shares no storage with any other binding, as asserted by the planner.
    Disjoint,
    /// Is the sole writer during the invocation, as asserted by the planner.
    Exclusive,
}

/// Declaration of one tensor argument of a program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingDecl {
    /// Representation metadata. For dense bindings, `layout` below owns the
    /// invocation layout contract; use `value_spec()` to query it. Packed storage
    /// retains the format-specific layout constraint recorded here.
    pub storage: StorageMetadata,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
    pub layout: BindingLayoutPolicy,
    pub aliasing: BindingAliasing,
}

impl BindingDecl {
    /// Caller-visible value contract derived from the binding policy.
    ///
    /// `Contiguous` permits nonzero offsets and `Canonicalize` accepts arbitrary
    /// valid dense strides. Their full policies remain in `self.layout`; the
    /// borrowed storage view conservatively reports `Unconstrained`. A
    /// canonicalization target is an execution layout, not an input requirement.
    pub fn value_spec(&self) -> ValueSpec<'_> {
        let mut storage = self.storage.as_spec();
        if storage.representation == StorageRepresentation::Dense {
            storage.layout_constraint = match &self.layout {
                BindingLayoutPolicy::Require(LayoutConstraint::Exact(layout)) => {
                    LayoutConstraintSpec::DenseStrided(layout)
                }
                BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous) => {
                    LayoutConstraintSpec::Canonical
                }
                BindingLayoutPolicy::Require(
                    LayoutConstraint::Contiguous | LayoutConstraint::AnyStrided,
                )
                | BindingLayoutPolicy::Canonicalize { .. } => LayoutConstraintSpec::Unconstrained,
            };
        }
        ValueSpec {
            semantic_dtype: self.dtype,
            logical_shape: &self.shape,
            storage,
        }
    }
}

/// Type of a scalar argument.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    Bool,
    U32,
    I64,
    F32,
    F64,
}

/// Scalar argument value. [`ScalarValue::scalar_type`] returns its
/// [`ScalarType`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ScalarValue {
    Bool(bool),
    U32(u32),
    I64(i64),
    F32(f32),
    F64(f64),
}

impl ScalarValue {
    /// The [`ScalarType`] this value belongs to.
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

/// Named declaration of one scalar argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScalarDecl {
    pub name: String,
    pub scalar_type: ScalarType,
}

/// Declared domain of a bounded `u64` or a length- and element-bounded
/// `u32` array.
///
/// Bounds belong to the declaration, not the value, so the compiler can
/// specialize on them. A declaration with `min > max` returns
/// [`RuntimeValueError::InvalidDeclaration`].
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

/// Named declaration of one runtime value argument.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeValueDecl {
    pub name: String,
    pub kind: RuntimeValueKind,
}

impl RuntimeValueDecl {
    /// Declares a `u64` runtime value in `min..=max`.
    pub fn u64(name: impl Into<String>, min: u64, max: u64) -> Self {
        Self {
            name: name.into(),
            kind: RuntimeValueKind::U64 { min, max },
        }
    }

    /// Declares a `u32` array of at most `max_len` elements. Elements may
    /// span the full `u32` range.
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

    /// Checks `value` against the declared kind and bounds.
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

/// A runtime value passed at invocation time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeValue {
    U64(u64),
    U32Array(Box<[u32]>),
}

/// Exact number of RNG counters an invocation must supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RngDecl {
    pub counter_count: u32,
}

/// Per-invocation RNG seed, nonce, and exactly [`RngDecl::counter_count`]
/// counters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngInvocation {
    pub seed: u64,
    pub nonce: u64,
    pub counters: Box<[u64]>,
}

/// Scalar, runtime-value, and RNG declarations in positional order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InvocationSignature {
    pub scalars: Vec<ScalarDecl>,
    pub runtime_values: Vec<RuntimeValueDecl>,
    pub rng: Option<RngDecl>,
}

/// Declared shape, dtype, and placement of one program output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputSignature {
    pub storage: StorageMetadata,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
}

impl OutputSignature {
    pub fn value_spec(&self) -> ValueSpec<'_> {
        ValueSpec {
            semantic_dtype: self.dtype,
            logical_shape: &self.shape,
            storage: self.storage.as_spec(),
        }
    }
}

/// Calling convention for a compiled program, including positional tensor
/// bindings, invocation parameters, and outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ProgramSignature {
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub outputs: Vec<OutputSignature>,
}

/// Arguments for one program call. Their positions match the
/// [`ProgramSignature`].
#[derive(Debug, Clone, Default)]
pub struct Invocation {
    pub bindings: Vec<ErasedBuffer>,
    pub scalars: Vec<ScalarValue>,
    pub runtime_values: Vec<RuntimeValue>,
    pub rng: Option<RngInvocation>,
}

impl ProgramSignature {
    /// Checks argument counts only (bindings, scalars, runtime values, RNG
    /// presence and counter count) without inspecting any buffer metadata.
    pub fn validate_invocation_counts(
        &self,
        bindings: usize,
        scalars: usize,
        runtime_values: usize,
        rng_counters: Option<usize>,
    ) -> Result<(), InvocationError> {
        validate_count("bindings", self.bindings.len(), bindings)?;
        validate_count("scalars", self.invocation.scalars.len(), scalars)?;
        validate_count(
            "runtime values",
            self.invocation.runtime_values.len(),
            runtime_values,
        )?;
        match (self.invocation.rng, rng_counters) {
            (None, None) => Ok(()),
            (Some(decl), Some(actual)) if actual == decl.counter_count as usize => Ok(()),
            (Some(decl), Some(actual)) => Err(InvocationError::RngCounterCount {
                expected: decl.counter_count as usize,
                actual,
            }),
            (expected, actual) => Err(InvocationError::RngPresence {
                expected: expected.is_some(),
                actual: actual.is_some(),
            }),
        }
    }

    /// Checks binding `binding`'s logical dtype/shape, representation, placement,
    /// and physical layout policy without requiring an [`ErasedBuffer`]. The
    /// handle constructor must already have checked the observed physical dtype,
    /// base alignment, and allocation extent with `ValueSpec::validate_buffer`
    /// and the backend's allocation checks.
    pub fn validate_binding_metadata(
        &self,
        binding: usize,
        value: ValueSpec<'_>,
        placement: &Placement,
        layout: &Layout,
    ) -> Result<(), InvocationError> {
        let Some(decl) = self.bindings.get(binding) else {
            return Err(InvocationError::Count {
                kind: "bindings",
                expected: self.bindings.len(),
                actual: binding.saturating_add(1),
            });
        };
        let dtype = value.semantic_dtype;
        if dtype != decl.dtype {
            return Err(InvocationError::DTypeMismatch {
                binding,
                expected: decl.dtype,
                actual: dtype,
            });
        }
        if placement != &decl.placement {
            return Err(InvocationError::PlacementMismatch { binding });
        }
        if value.logical_shape != decl.shape {
            return Err(InvocationError::ShapeMismatch { binding });
        }
        if value.storage.representation != decl.storage.representation {
            return Err(InvocationError::RepresentationMismatch {
                binding,
                expected: decl.storage.representation,
                actual: value.storage.representation,
            });
        }
        let layout_matches = match &decl.layout {
            BindingLayoutPolicy::Require(LayoutConstraint::Exact(expected)) => layout == expected,
            BindingLayoutPolicy::Require(LayoutConstraint::Contiguous) => layout.is_contiguous(),
            BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous) => {
                layout.is_contiguous() && layout.offset() == 0
            }
            BindingLayoutPolicy::Require(LayoutConstraint::AnyStrided)
            | BindingLayoutPolicy::Canonicalize { .. } => true,
        };
        if !layout_matches {
            return Err(InvocationError::LayoutMismatch { binding });
        }
        value
            .validate()
            .map_err(|reason| InvocationError::InvalidStorage { binding, reason })?;
        let expected = decl.value_spec();
        let geometry = value
            .canonical_geometry()
            .map_err(|reason| InvocationError::InvalidStorage { binding, reason })?;
        let extent = layout
            .checked_byte_size(geometry.physical_dtype)
            .ok_or_else(|| InvocationError::InvalidStorage {
                binding,
                reason: "physical layout byte extent overflows".to_string(),
            })?;
        value
            .validate_buffer(geometry.physical_dtype, layout, extent)
            .map_err(|reason| InvocationError::InvalidStorage { binding, reason })?;
        expected
            .validate_buffer(geometry.physical_dtype, layout, extent)
            .map_err(|reason| InvocationError::InvalidStorage { binding, reason })?;
        if let BindingLayoutPolicy::Canonicalize { target } = &decl.layout {
            let mut target_spec = expected;
            if target_spec.storage.representation == StorageRepresentation::Dense {
                target_spec.storage.layout_constraint = LayoutConstraintSpec::DenseStrided(target);
            }
            let target_extent = target
                .checked_byte_size(geometry.physical_dtype)
                .ok_or_else(|| InvocationError::InvalidStorage {
                    binding,
                    reason: "canonicalization target byte extent overflows".to_string(),
                })?;
            target_spec
                .validate_buffer(geometry.physical_dtype, target, target_extent)
                .map_err(|reason| InvocationError::InvalidStorage { binding, reason })?;
        }
        Ok(())
    }

    /// Checks the type of scalar argument `scalar`.
    pub fn validate_scalar_metadata(
        &self,
        scalar: usize,
        scalar_type: ScalarType,
    ) -> Result<(), InvocationError> {
        let Some(decl) = self.invocation.scalars.get(scalar) else {
            return Err(InvocationError::Count {
                kind: "scalars",
                expected: self.invocation.scalars.len(),
                actual: scalar.saturating_add(1),
            });
        };
        if decl.scalar_type != scalar_type {
            return Err(InvocationError::ScalarTypeMismatch { scalar });
        }
        Ok(())
    }

    /// Checks `runtime_value` against its declaration and adds its argument
    /// position to any [`RuntimeValueError`].
    pub fn validate_runtime_value_metadata(
        &self,
        runtime_value: usize,
        value: &RuntimeValue,
    ) -> Result<(), InvocationError> {
        let Some(decl) = self.invocation.runtime_values.get(runtime_value) else {
            return Err(InvocationError::Count {
                kind: "runtime values",
                expected: self.invocation.runtime_values.len(),
                actual: runtime_value.saturating_add(1),
            });
        };
        decl.validate(value)
            .map_err(|source| InvocationError::RuntimeValue {
                runtime_value,
                source,
            })
    }

    /// Checks counts, buffer ownership by `runtime`, binding metadata, scalar
    /// types, and runtime-value bounds, in that order.
    pub fn validate_invocation(
        &self,
        runtime: RuntimeId,
        invocation: &Invocation,
    ) -> Result<(), InvocationError> {
        self.validate_invocation_counts(
            invocation.bindings.len(),
            invocation.scalars.len(),
            invocation.runtime_values.len(),
            invocation.rng.as_ref().map(|rng| rng.counters.len()),
        )?;

        for (index, buffer) in invocation.bindings.iter().enumerate() {
            if buffer.runtime_id() != runtime {
                return Err(InvocationError::InvalidOwner {
                    binding: index,
                    expected: runtime,
                    actual: buffer.runtime_id(),
                });
            }
            self.validate_binding_metadata(
                index,
                buffer.value_spec(),
                buffer.placement(),
                buffer.layout(),
            )?;
        }

        for (index, value) in invocation.scalars.iter().enumerate() {
            self.validate_scalar_metadata(index, value.scalar_type())?;
        }

        for (index, value) in invocation.runtime_values.iter().enumerate() {
            self.validate_runtime_value_metadata(index, value)?;
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

/// Why a single runtime value failed validation against its declaration.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeValueError {
    /// The declaration itself is inconsistent (`min > max`).
    InvalidDeclaration { name: String },
    /// The value's kind does not match the declared kind.
    KindMismatch { name: String },
    /// A scalar value outside its declared `min..=max` range.
    OutOfBounds {
        name: String,
        value: u64,
        min: u64,
        max: u64,
    },
    /// An array longer than its declared `max_len`.
    TooLong {
        name: String,
        len: usize,
        max_len: usize,
    },
    /// An array element outside its declared `min..=max` range.
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

/// Why an [`Invocation`] failed validation against a [`ProgramSignature`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InvocationError {
    /// Wrong number of positional arguments of a kind (`"bindings"`,
    /// `"scalars"` or `"runtime values"`).
    Count {
        kind: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A different runtime owns the binding buffer.
    InvalidOwner {
        binding: usize,
        expected: RuntimeId,
        actual: RuntimeId,
    },
    /// A binding's dtype differs from the declaration.
    DTypeMismatch {
        binding: usize,
        expected: DType,
        actual: DType,
    },
    /// A binding's represented values use a different storage format.
    RepresentationMismatch {
        binding: usize,
        expected: StorageRepresentation,
        actual: StorageRepresentation,
    },
    /// Invalid representation geometry or layout at the boundary.
    InvalidStorage { binding: usize, reason: String },
    /// A binding's placement differs from the declaration.
    PlacementMismatch { binding: usize },
    /// A binding's shape differs from the declaration.
    ShapeMismatch { binding: usize },
    /// A binding's layout violates the declared policy.
    LayoutMismatch { binding: usize },
    /// A scalar's type differs from the declaration.
    ScalarTypeMismatch { scalar: usize },
    /// A runtime value failed its declaration's checks.
    RuntimeValue {
        runtime_value: usize,
        source: RuntimeValueError,
    },
    /// The invocation and declaration disagree on whether RNG state is present.
    RngPresence { expected: bool, actual: bool },
    /// The invocation supplied the wrong number of RNG counters.
    RngCounterCount { expected: usize, actual: usize },
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
            InvocationError::RepresentationMismatch {
                binding,
                expected,
                actual,
            } => write!(
                f,
                "binding {binding} has representation {actual:?}, expected {expected:?}"
            ),
            InvocationError::InvalidStorage { binding, reason } => {
                write!(f, "binding {binding} has invalid storage: {reason}")
            }
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

        fn value_spec(&self) -> ValueSpec<'_> {
            dense_view(self.dtype, self.layout.shape(), &self.layout)
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
                storage: StorageMetadata::dense(),
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

    #[test]
    fn count_and_metadata_validation_do_not_require_erased_buffers() {
        let mut signature = signature();
        signature.bindings[0].layout =
            BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous);
        assert!(signature.validate_invocation_counts(1, 0, 1, None).is_ok());
        assert_eq!(
            signature.validate_invocation_counts(0, 0, 1, None),
            Err(InvocationError::Count {
                kind: "bindings",
                expected: 1,
                actual: 0,
            })
        );

        let placement = Placement::new(DeviceId::new("cpu:0"));
        assert!(signature
            .validate_binding_metadata(
                0,
                ValueSpec::dense(DType::F32, &[2]),
                &placement,
                &Layout::contiguous(vec![2]),
            )
            .is_ok());
        assert_eq!(
            signature.validate_binding_metadata(
                0,
                ValueSpec::dense(DType::F32, &[2]),
                &placement,
                &Layout::new(vec![2], vec![1], 1),
            ),
            Err(InvocationError::LayoutMismatch { binding: 0 })
        );
        assert!(signature
            .validate_runtime_value_metadata(0, &RuntimeValue::U64(2))
            .is_ok());
    }

    fn dense_binding(layout: BindingLayoutPolicy) -> ProgramSignature {
        ProgramSignature {
            bindings: vec![BindingDecl {
                shape: vec![2, 3],
                dtype: DType::F32,
                storage: StorageMetadata::dense(),
                placement: Placement::new(DeviceId::new("cpu:0")),
                layout,
                aliasing: BindingAliasing::MayAlias,
            }],
            ..ProgramSignature::default()
        }
    }

    fn dense_view<'a>(dtype: DType, shape: &'a [usize], layout: &'a Layout) -> ValueSpec<'a> {
        ValueSpec {
            semantic_dtype: dtype,
            logical_shape: shape,
            storage: crate::StorageSpec {
                representation: StorageRepresentation::Dense,
                layout_constraint: LayoutConstraintSpec::DenseStrided(layout),
            },
        }
    }

    #[test]
    fn dense_binding_policy_owns_strided_and_offset_rebinding() {
        let owner = RuntimeId::new();
        let layouts = [
            Layout::contiguous(vec![2, 3]),
            Layout::new(vec![2, 3], vec![3, 1], 2),
            Layout::new(vec![2, 3], vec![1, 2], 0),
            Layout::new(vec![2, 3], vec![8, 2], 1),
            Layout::new(vec![2, 3], vec![0, 1], 0),
        ];
        let policies = [
            (
                BindingLayoutPolicy::Require(LayoutConstraint::AnyStrided),
                [true, true, true, true, true],
            ),
            (
                BindingLayoutPolicy::Canonicalize {
                    target: layouts[0].clone(),
                },
                [true, true, true, true, true],
            ),
            (
                BindingLayoutPolicy::Require(LayoutConstraint::Contiguous),
                [true, true, false, false, false],
            ),
            (
                BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous),
                [true, false, false, false, false],
            ),
            (
                BindingLayoutPolicy::Require(LayoutConstraint::Exact(layouts[3].clone())),
                [false, false, false, true, false],
            ),
        ];
        for (policy, accepted) in policies {
            let signature = dense_binding(policy.clone());
            for (layout, accepted) in layouts.iter().zip(accepted) {
                let invocation = Invocation {
                    bindings: vec![ErasedBuffer::new(TestBuffer {
                        owner,
                        placement: signature.bindings[0].placement.clone(),
                        dtype: DType::F32,
                        layout: layout.clone(),
                    })],
                    ..Invocation::default()
                };
                let result = signature.validate_invocation(owner, &invocation);
                if accepted {
                    result.unwrap();
                } else {
                    assert_eq!(
                        result,
                        Err(InvocationError::LayoutMismatch { binding: 0 }),
                        "{policy:?} with {layout:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn canonicalization_validates_its_target_separately_from_the_input() {
        let source = Layout::new(vec![2, 3], vec![1, 2], 0);
        let value = dense_view(DType::F32, &[2, 3], &source);
        for target in [
            Layout::contiguous(vec![2, 3]),
            Layout::new(vec![2, 3], vec![5, 1], 1),
        ] {
            let signature = dense_binding(BindingLayoutPolicy::Canonicalize { target });
            signature
                .validate_binding_metadata(0, value, &signature.bindings[0].placement, &source)
                .unwrap();
        }
        for target in [
            Layout::contiguous(vec![3, 2]),
            Layout::new(vec![2, 3], vec![usize::MAX, 1], 0),
        ] {
            let signature = dense_binding(BindingLayoutPolicy::Canonicalize { target });
            assert!(matches!(
                signature.validate_binding_metadata(
                    0,
                    value,
                    &signature.bindings[0].placement,
                    &source
                ),
                Err(InvocationError::InvalidStorage { .. })
            ));
        }
    }

    #[test]
    fn permissive_dense_policies_still_require_truthful_value_metadata() {
        let physical = Layout::new(vec![2, 3], vec![1, 2], 0);
        let value = dense_view(DType::F32, &[2, 3], &physical);
        // Physical scalar type is checked before handle publication; permissive
        // binding layouts do not authorize changing representation or dtype.
        assert!(value
            .validate_buffer(DType::F16, &physical, 24)
            .unwrap_err()
            .contains("physical dtype"));
        value.validate_buffer(DType::F32, &physical, 24).unwrap();
        for policy in [
            BindingLayoutPolicy::Require(LayoutConstraint::AnyStrided),
            BindingLayoutPolicy::Canonicalize {
                target: Layout::contiguous(vec![2, 3]),
            },
        ] {
            let signature = dense_binding(policy);
            let placement = &signature.bindings[0].placement;
            assert_eq!(
                signature.validate_binding_metadata(
                    0,
                    ValueSpec {
                        semantic_dtype: DType::F16,
                        ..value
                    },
                    placement,
                    &physical
                ),
                Err(InvocationError::DTypeMismatch {
                    binding: 0,
                    expected: DType::F32,
                    actual: DType::F16
                })
            );
            assert_eq!(
                signature.validate_binding_metadata(
                    0,
                    ValueSpec {
                        logical_shape: &[3, 2],
                        ..value
                    },
                    placement,
                    &physical
                ),
                Err(InvocationError::ShapeMismatch { binding: 0 })
            );
            assert!(matches!(
                signature.validate_binding_metadata(
                    0,
                    ValueSpec::dense(DType::F32, &[2, 3]),
                    placement,
                    &physical
                ),
                Err(InvocationError::InvalidStorage { .. })
            ));
            assert!(matches!(
                signature.validate_binding_metadata(
                    0,
                    value,
                    placement,
                    &Layout::contiguous(vec![2, 3])
                ),
                Err(InvocationError::InvalidStorage { .. })
            ));
        }
    }

    #[test]
    fn permissive_binding_policies_cannot_weaken_packed_canonical_layout() {
        let storage = StorageMetadata::packed(crate::GgmlKQuant::Q4K);
        let value = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[2, 256],
            storage: storage.as_spec(),
        };
        let canonical = Layout::contiguous(vec![2, 144]);
        assert!(value.validate_buffer(DType::F32, &canonical, 288).is_err());
        for policy in [
            BindingLayoutPolicy::Require(LayoutConstraint::AnyStrided),
            BindingLayoutPolicy::Canonicalize {
                target: canonical.clone(),
            },
        ] {
            let mut signature = dense_binding(policy);
            signature.bindings[0].shape = vec![2, 256];
            signature.bindings[0].storage = storage.clone();
            let placement = &signature.bindings[0].placement;
            signature
                .validate_binding_metadata(0, value, placement, &canonical)
                .unwrap();
            for layout in [
                Layout::new(vec![2, 144], vec![144, 1], 1),
                Layout::new(vec![2, 144], vec![145, 1], 0),
            ] {
                assert!(matches!(
                    signature.validate_binding_metadata(0, value, placement, &layout),
                    Err(InvocationError::InvalidStorage { .. })
                ));
            }
            let unconstrained = ValueSpec {
                storage: crate::StorageSpec {
                    layout_constraint: LayoutConstraintSpec::Unconstrained,
                    ..value.storage
                },
                ..value
            };
            assert!(matches!(
                signature.validate_binding_metadata(0, unconstrained, placement, &canonical),
                Err(InvocationError::InvalidStorage { .. })
            ));
        }
        let mut signature = dense_binding(BindingLayoutPolicy::Canonicalize {
            target: Layout::new(vec![2, 144], vec![144, 1], 1),
        });
        signature.bindings[0].shape = vec![2, 256];
        signature.bindings[0].storage = storage;
        let value = signature.bindings[0].value_spec();
        assert!(matches!(
            signature.validate_binding_metadata(
                0,
                value,
                &signature.bindings[0].placement,
                &canonical
            ),
            Err(InvocationError::InvalidStorage { .. })
        ));
    }

    #[test]
    fn invocation_checks_packed_identity_and_logical_geometry() {
        use crate::{GgmlKQuant, StorageMetadata};
        #[derive(Debug)]
        struct PackedBuffer {
            owner: RuntimeId,
            placement: Placement,
            layout: Layout,
            storage: StorageMetadata,
        }
        impl Buffer for PackedBuffer {
            fn runtime_id(&self) -> RuntimeId {
                self.owner
            }
            fn placement(&self) -> &Placement {
                &self.placement
            }
            fn dtype(&self) -> DType {
                DType::F32
            }
            fn layout(&self) -> &Layout {
                &self.layout
            }
            fn value_spec(&self) -> ValueSpec<'_> {
                ValueSpec {
                    semantic_dtype: DType::F32,
                    logical_shape: &[2, 256],
                    storage: self.storage.as_spec(),
                }
            }
            fn as_any(&self) -> &dyn Any {
                self
            }
        }
        let owner = RuntimeId::new();
        let placement = Placement::new(DeviceId::new("cpu:0"));
        let storage = StorageMetadata::packed(GgmlKQuant::Q4K);
        let signature = ProgramSignature {
            bindings: vec![BindingDecl {
                shape: vec![2, 256],
                dtype: DType::F32,
                storage: storage.clone(),
                placement: placement.clone(),
                layout: BindingLayoutPolicy::Require(LayoutConstraint::ZeroOffsetContiguous),
                aliasing: BindingAliasing::MayAlias,
            }],
            ..ProgramSignature::default()
        };
        let invoke = |codec: GgmlKQuant| Invocation {
            bindings: vec![ErasedBuffer::new(PackedBuffer {
                owner,
                placement: placement.clone(),
                layout: Layout::contiguous(vec![2, codec.block_bytes()]),
                storage: StorageMetadata::packed(codec),
            })],
            ..Invocation::default()
        };
        signature
            .validate_invocation(owner, &invoke(GgmlKQuant::Q4K))
            .unwrap();
        assert!(matches!(
            signature.validate_invocation(owner, &invoke(GgmlKQuant::Q5K)),
            Err(InvocationError::RepresentationMismatch { binding: 0, .. })
        ));
        let physical = Layout::contiguous(vec![2, 144]);
        assert!(matches!(
            signature.validate_binding_metadata(
                0,
                ValueSpec::dense(DType::F32, &[2, 256]),
                &placement,
                &physical
            ),
            Err(InvocationError::RepresentationMismatch { .. })
        ));
        assert!(matches!(
            signature.validate_binding_metadata(
                0,
                ValueSpec::dense(DType::U8, &[2, 144]),
                &placement,
                &physical
            ),
            Err(InvocationError::DTypeMismatch { .. })
        ));
        let value = ValueSpec {
            semantic_dtype: DType::F32,
            logical_shape: &[2, 256],
            storage: storage.as_spec(),
        };
        assert!(matches!(
            signature.validate_binding_metadata(
                0,
                value,
                &placement,
                &Layout::contiguous(vec![2, 143])
            ),
            Err(InvocationError::InvalidStorage { .. })
        ));
        assert!(matches!(
            signature.validate_binding_metadata(
                0,
                ValueSpec {
                    logical_shape: &[1, 512],
                    ..value
                },
                &placement,
                &physical
            ),
            Err(InvocationError::ShapeMismatch { .. })
        ));
        let mut other = signature.clone();
        other.bindings[0].storage = StorageMetadata::packed(GgmlKQuant::Q5K);
        assert_ne!(signature, other);
    }
}
