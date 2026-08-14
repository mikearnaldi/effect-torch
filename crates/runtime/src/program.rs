//! Program signatures and invocation validation.
//!
//! A compiled program declares its calling convention as a
//! [`ProgramSignature`]: the tensor [`BindingDecl`]s it reads, the scalar
//! and [`RuntimeValueDecl`] parameters it accepts per call, an optional
//! [`RngDecl`], and its [`OutputSignature`]s. A call site assembles an
//! [`Invocation`] and the runtime validates it against the signature via
//! [`ProgramSignature::validate_invocation`] *before* any execution.
//!
//! # Validation rules
//!
//! [`validate_invocation`](ProgramSignature::validate_invocation) checks,
//! in order: argument counts (bindings, scalars, runtime values, RNG
//! presence and counter count); buffer ownership (every binding must carry
//! the calling runtime's [`RuntimeId`]); per-binding dtype, placement,
//! shape and layout policy; scalar types; and runtime-value bounds
//! (declared `min..=max` ranges, array length caps and element ranges).
//! Each failure is a distinct [`InvocationError`] variant.
//!
//! # Layout policies
//!
//! Each binding declares a [`BindingLayoutPolicy`]: either
//! `Require(`[`LayoutConstraint`]`)` — the caller's buffer must satisfy
//! the constraint (`Exact`, `Contiguous`, `ZeroOffsetContiguous` or
//! `AnyStrided`) — or `Canonicalize { target }`, which accepts any layout
//! at validation time because the backend will canonicalize the buffer to
//! `target` before use. [`BindingAliasing`] records whether a binding may
//! share storage with other bindings (`MayAlias`), provably does not
//! (`Disjoint`), or is the sole writer of its storage (`Exclusive`).
//!
//! # RNG
//!
//! Randomness is explicit and replayable: [`RngDecl`] fixes the counter
//! count at compile time and each [`RngInvocation`] supplies a seed, a
//! nonce and exactly that many counters, so a frozen graph never replays
//! stale random state.

use crate::{DType, ErasedBuffer, Layout, Placement, RuntimeId};
use std::error::Error;
use std::fmt;

/// Requirement imposed on a binding's layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LayoutConstraint {
    /// The layout must equal this exact shape/strides/offset triple.
    Exact(Layout),
    /// Densely packed row-major (any offset).
    Contiguous,
    /// Densely packed row-major with offset 0.
    ZeroOffsetContiguous,
    /// Any strides are acceptable.
    AnyStrided,
}

/// How a binding's layout is treated at the call boundary.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BindingLayoutPolicy {
    /// The caller's buffer must already satisfy the constraint.
    Require(LayoutConstraint),
    /// Any layout is accepted; the backend canonicalizes to `target`.
    Canonicalize { target: Layout },
}

/// Planner/backend assertion about whether a binding's storage may overlap
/// other storage.
///
/// [`ProgramSignature::validate_invocation`] records but does not prove these
/// relationships: invocation validation receives independent buffer handles,
/// not a portable byte-range alias model. Backends that rely on `Disjoint` or
/// `Exclusive` must enforce the assertion at their ownership boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BindingAliasing {
    /// May share storage with other bindings.
    MayAlias,
    /// Asserted to share no storage with any other binding.
    Disjoint,
    /// Asserted to be the sole writer for the duration of the invocation.
    Exclusive,
}

/// Declaration of one tensor argument of a program.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindingDecl {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
    pub layout: BindingLayoutPolicy,
    pub aliasing: BindingAliasing,
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

/// Value of a scalar argument; [`ScalarValue::scalar_type`] reports the
/// matching [`ScalarType`].
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

/// Declared domain of a runtime value: a bounded `u64` or a length-capped,
/// element-bounded `u32` array.
///
/// Bounds are part of the *declaration* (not the value) so the compiler can
/// specialize on them; a declaration with `min > max` is itself invalid and
/// reported as [`RuntimeValueError::InvalidDeclaration`].
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
    /// Declares a `u64` runtime value accepted in `min..=max`.
    pub fn u64(name: impl Into<String>, min: u64, max: u64) -> Self {
        Self {
            name: name.into(),
            kind: RuntimeValueKind::U64 { min, max },
        }
    }

    /// Declares a `u32` array runtime value of at most `max_len` elements
    /// spanning the full `u32` range.
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

    /// Validates `value` against the declared kind and bounds.
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

/// A runtime value supplied at invocation time.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum RuntimeValue {
    U64(u64),
    U32Array(Box<[u32]>),
}

/// Declared RNG requirement of a program: the exact number of counters an
/// invocation must supply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RngDecl {
    pub counter_count: u32,
}

/// Per-invocation RNG state: seed, nonce and exactly
/// [`RngDecl::counter_count`] counters.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RngInvocation {
    pub seed: u64,
    pub nonce: u64,
    pub counters: Box<[u64]>,
}

/// The non-tensor part of a signature: scalar, runtime-value and RNG
/// declarations, in positional order.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct InvocationSignature {
    pub scalars: Vec<ScalarDecl>,
    pub runtime_values: Vec<RuntimeValueDecl>,
    pub rng: Option<RngDecl>,
}

/// Declared shape, dtype and placement of one program output.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct OutputSignature {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub placement: Placement,
}

/// Full calling convention of a compiled program: positional tensor
/// bindings, invocation parameters and outputs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct ProgramSignature {
    pub bindings: Vec<BindingDecl>,
    pub invocation: InvocationSignature,
    pub outputs: Vec<OutputSignature>,
}

/// Arguments for one call of a program, positionally matched against the
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

    /// Validates the metadata of binding `binding` (dtype, placement,
    /// shape and layout policy) without an [`ErasedBuffer`], so callers
    /// that track metadata separately can still check conformance.
    pub fn validate_binding_metadata(
        &self,
        binding: usize,
        dtype: DType,
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
        if layout.shape() != decl.shape {
            return Err(InvocationError::ShapeMismatch { binding });
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
        Ok(())
    }

    /// Validates the type of scalar argument `scalar`.
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

    /// Validates runtime value `runtime_value` against its declaration,
    /// wrapping any [`RuntimeValueError`] with the argument's position.
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

    /// Full validation of an invocation against this signature: counts,
    /// buffer ownership by `runtime`, binding metadata, scalar types and
    /// runtime-value bounds, in that order.
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
                buffer.dtype(),
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
    /// A binding buffer is owned by a different runtime.
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
    /// RNG state was supplied but not declared, or vice versa.
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
            .validate_binding_metadata(0, DType::F32, &placement, &Layout::contiguous(vec![2]),)
            .is_ok());
        assert_eq!(
            signature.validate_binding_metadata(
                0,
                DType::F32,
                &placement,
                &Layout::new(vec![2], vec![1], 1),
            ),
            Err(InvocationError::LayoutMismatch { binding: 0 })
        );
        assert!(signature
            .validate_runtime_value_metadata(0, &RuntimeValue::U64(2))
            .is_ok());
    }
}
