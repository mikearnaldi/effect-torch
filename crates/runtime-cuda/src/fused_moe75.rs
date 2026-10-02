//! Opt-in model-specific FlashInfer MoE bridge. Canonical tensor storage stays unchanged.
use crate::{buffer::CudaBuffer, CudaDevice, CudaValue};
use cudarc::driver::{CudaContext, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};
use effect_torch_compiler::{DenseNodeId, NativeRegion};
use effect_torch_runtime::{DType, StorageRepresentation};
use libloading::Library;
use std::{
    ffi::{c_char, c_int, c_void},
    path::PathBuf,
    ptr,
    sync::{Arc, LazyLock, Mutex, Weak},
};

const GATE_BYTES: usize = 128 * 1408 * 2816 * 2;
const DOWN_BYTES: usize = 128 * 2816 * 704 * 2;
const ERROR_BYTES: usize = 2048;
type Create = unsafe extern "C" fn(
    u32,
    c_int,
    c_int,
    *mut *mut c_void,
    *mut usize,
    *mut usize,
    *mut c_int,
    *mut c_int,
    *mut c_char,
    usize,
) -> c_int;
type Pack =
    unsafe extern "C" fn(*const c_void, *mut c_void, *mut c_void, *mut c_char, usize) -> c_int;
type Run = unsafe extern "C" fn(
    *mut c_void,
    u32,
    *const c_void,
    *const i32,
    *const f32,
    *const c_void,
    *const c_void,
    *mut c_void,
    *mut c_void,
    usize,
    *mut c_void,
    usize,
    *mut c_void,
    *mut c_char,
    usize,
) -> c_int;
type Destroy = unsafe extern "C" fn(*mut c_void, *mut c_char, usize) -> c_int;
struct Api {
    path: PathBuf,
    _library: Library,
    create: Create,
    pack: Pack,
    run: Run,
    destroy: Destroy,
}
impl Api {
    fn load(path: PathBuf) -> Result<Arc<Self>, String> {
        // The explicitly configured library must implement the frozen bridge.h ABI.
        unsafe {
            let library = Library::new(&path).map_err(|e| e.to_string())?;
            let create = *library
                .get::<Create>(b"et_fi73_create\0")
                .map_err(|e| e.to_string())?;
            let pack = *library
                .get::<Pack>(b"et_fi73_pack\0")
                .map_err(|e| e.to_string())?;
            let run = *library
                .get::<Run>(b"et_fi73_run\0")
                .map_err(|e| e.to_string())?;
            let destroy = *library
                .get::<Destroy>(b"et_fi73_destroy\0")
                .map_err(|e| e.to_string())?;
            Ok(Arc::new(Self {
                path,
                _library: library,
                create,
                pack,
                run,
                destroy,
            }))
        }
    }
}
pub(crate) fn enabled() -> bool {
    std::env::var("EFFECT_TORCH_CUDA_FUSED_MOE75").as_deref() == Ok("1")
}
/// Unique TopK IDs make the proved stable rank expression a permutation,
/// independently of its explicit tie positions. Arbitrary scatter ranks cannot
/// be omitted by the native weighted finalizer.
pub(crate) fn proves_route_ranks(
    region: Option<&NativeRegion>,
    ranks: DenseNodeId,
    experts: DenseNodeId,
    tokens: usize,
) -> bool {
    matches!(region, Some(NativeRegion::ExpertRouteRank(rank))
        if rank.output == ranks
            && rank.rows == tokens
            && rank.routes == 8
            && rank.inputs.first() == Some(&experts))
}
fn parse_tactic(value: Option<&str>) -> Result<i32, String> {
    let Some(value) = value else { return Ok(-1) };
    let tactic = value
        .parse::<i32>()
        .map_err(|_| "expected an integer tactic index".to_string())?;
    if tactic < -1 {
        return Err("tactic must be -1 or nonnegative".into());
    }
    Ok(tactic)
}
fn configured_tactic(name: &str) -> Result<i32, String> {
    match std::env::var(name) {
        Ok(value) => parse_tactic(Some(&value)).map_err(|e| format!("{name}: {e}")),
        Err(std::env::VarError::NotPresent) => Ok(-1),
        Err(error) => Err(format!("{name}: {error}")),
    }
}
fn checked(code: c_int, error: &[c_char; ERROR_BYTES], operation: &str) -> Result<(), String> {
    if code == 0 {
        return Ok(());
    }
    let bytes: Vec<u8> = error
        .iter()
        .take_while(|&&b| b != 0)
        .map(|&b| b as u8)
        .collect();
    let message = String::from_utf8_lossy(&bytes);
    Err(format!("fused MoE75 {operation}: {message}"))
}
struct Packed {
    source: Arc<CudaBuffer<u8>>,
    buffer: CudaBuffer<u8>,
    _api: Arc<Api>,
    guard: Arc<RouteGuard>,
}
struct RouteGuard {
    context: Arc<CudaContext>,
    function: CudaFunction,
}
static GUARDS: LazyLock<Mutex<Vec<Weak<RouteGuard>>>> = LazyLock::new(Default::default);
fn route_guard(context: &Arc<CudaContext>) -> Result<Arc<RouteGuard>, String> {
    let mut entries = GUARDS
        .lock()
        .map_err(|_| "fused MoE75 guard cache poisoned")?;
    entries.retain(|entry| entry.strong_count() > 0);
    if let Some(guard) = entries
        .iter()
        .filter_map(Weak::upgrade)
        .find(|guard| Arc::ptr_eq(&guard.context, context))
    {
        return Ok(guard);
    }
    let ptx = compile_ptx_with_opts(
        include_str!("kernels/fused_moe75_guard.cu"),
        CompileOptions {
            arch: Some("compute_120"),
            name: Some("fused_moe75_guard.cu".into()),
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let module = context.load_module(ptx).map_err(|e| e.to_string())?;
    let function = module
        .load_function("et_fused_moe75_guard")
        .map_err(|e| e.to_string())?;
    let guard = Arc::new(RouteGuard {
        context: Arc::clone(context),
        function,
    });
    entries.push(Arc::downgrade(&guard));
    Ok(guard)
}
struct CacheEntry {
    source: Weak<CudaBuffer<u8>>,
    packed: Weak<Packed>,
    context: usize,
}
static PACKED: LazyLock<Mutex<Vec<CacheEntry>>> = LazyLock::new(Default::default);

fn same_source<T>(witness: &Weak<T>, source: &Arc<T>) -> bool {
    witness
        .upgrade()
        .is_some_and(|owner| Arc::ptr_eq(&owner, source))
}

struct Runner {
    handle: *mut c_void,
    api: Arc<Api>,
    stream: Arc<CudaStream>,
    retained: Arc<Packed>,
}
// Access is mutex-protected and CUDA context binding occurs on every operation.
unsafe impl Send for Runner {}
impl Drop for Runner {
    fn drop(&mut self) {
        if self.stream.context().bind_to_thread().is_err() || self.stream.synchronize().is_err() {
            // Never unload code or destroy mutable runner storage with outstanding work.
            std::mem::forget(Arc::clone(&self.api));
            std::mem::forget(Arc::clone(&self.stream));
            std::mem::forget(Arc::clone(&self.retained));
            eprintln!("fused MoE75 runner drain failed; quarantining runner and library");
            return;
        }
        let mut error = [0; ERROR_BYTES];
        let result = unsafe { (self.api.destroy)(self.handle, error.as_mut_ptr(), ERROR_BYTES) };
        if let Err(message) = checked(result, &error, "destroy") {
            eprintln!("{message}");
        }
    }
}
pub(crate) struct Plan {
    pub(crate) workspace_bytes: usize,
    pub(crate) map_bytes: usize,
    pub(crate) route_scratch_bytes: usize,
    pub(crate) selected_gemm1: i32,
    pub(crate) selected_gemm2: i32,
    tokens: usize,
    runner: Mutex<Runner>,
    packed: Arc<Packed>,
}

/// Only immutable captured model weights are eligible. The caller must exclude
/// bindings, planned/workspace values, and weights subsequently updated in place.
pub(crate) fn prepare(
    device: &Arc<CudaDevice>,
    source: &CudaValue,
    tokens: usize,
) -> Result<Arc<Plan>, String> {
    if !enabled() {
        return Err("fused MoE75 is disabled".into());
    }
    let gemm1 = configured_tactic("EFFECT_TORCH_CUDA_FUSED_MOE75_GEMM1")?;
    let gemm2 = configured_tactic("EFFECT_TORCH_CUDA_FUSED_MOE75_GEMM2")?;
    if !matches!(tokens, 64 | 256)
        || !Arc::ptr_eq(device, &source.device)
        || source.dtype() != DType::BF16
        || source.shape() != [128, 1408, 2816]
        || source.spec().storage.representation != StorageRepresentation::Dense
        || source.buffer.len() != GATE_BYTES
    {
        return Err("fused MoE75 captured weight geometry/device differs".into());
    }
    device
        .stream
        .context()
        .bind_to_thread()
        .map_err(|e| e.to_string())?;
    let library_path = std::fs::canonicalize(
        std::env::var_os("EFFECT_TORCH_CUDA_FUSED_MOE75_SO")
            .ok_or("fused MoE75 shared library path is missing")?,
    )
    .map_err(|e| format!("fused MoE75 library path: {e}"))?;
    // Compilation packing must not race an invocation or a capture on this device.
    let _execution = device
        .graph_execution
        .lock()
        .map_err(|_| "fused MoE75 device lock poisoned")?;
    let context = device.stream.context().cu_ctx() as usize;
    let mut entries = PACKED
        .lock()
        .map_err(|_| "fused MoE75 packed cache poisoned")?;
    entries.retain(|e| e.source.strong_count() > 0 && e.packed.strong_count() > 0);
    let cached = entries.iter().find_map(|entry| {
        (entry.context == context && same_source(&entry.source, &source.buffer))
            .then(|| entry.packed.upgrade())
            .flatten()
            .filter(|packed| packed._api.path == library_path)
    });
    let packed = if let Some(packed) = cached {
        packed
    } else {
        let api = Api::load(library_path)?;
        let allocation = device
            .stream
            .alloc_zeros::<u8>(GATE_BYTES)
            .map_err(|e| e.to_string())?;
        let packed = Arc::new(Packed {
            source: Arc::clone(&source.buffer),
            buffer: CudaBuffer::from_slice(allocation),
            _api: api,
            guard: route_guard(device.stream.context())?,
        });
        let mut error = [0; ERROR_BYTES];
        let result = unsafe {
            (packed._api.pack)(
                packed.source.address() as *const c_void,
                packed.buffer.address() as *mut c_void,
                device.stream.cu_stream().cast(),
                error.as_mut_ptr(),
                ERROR_BYTES,
            )
        };
        // Even a failed FFI call may have queued packing work. Publish only after a checked fence.
        if let Err(drain) = device.stream.synchronize() {
            std::mem::forget(packed);
            return Err(format!(
                "fused MoE75 packing drain failed; buffers quarantined: {drain}"
            ));
        }
        checked(result, &error, "pack")?;
        entries.push(CacheEntry {
            source: Arc::downgrade(&source.buffer),
            packed: Arc::downgrade(&packed),
            context,
        });
        packed
    };
    drop(entries);
    let api = Arc::clone(&packed._api);
    let mut handle = ptr::null_mut();
    let (mut workspace_bytes, mut map_bytes, mut first, mut second) = (0, 0, 0, 0);
    let mut error = [0; ERROR_BYTES];
    let result = unsafe {
        (api.create)(
            tokens as u32,
            gemm1,
            gemm2,
            &mut handle,
            &mut workspace_bytes,
            &mut map_bytes,
            &mut first,
            &mut second,
            error.as_mut_ptr(),
            ERROR_BYTES,
        )
    };
    checked(result, &error, "create")?;
    if handle.is_null() {
        return Err("fused MoE75 create returned a null runner".into());
    }
    let runner = Runner {
        handle,
        api,
        stream: Arc::clone(&device.stream),
        retained: Arc::clone(&packed),
    };
    if workspace_bytes == 0 || map_bytes < tokens * 8 * 4 {
        return Err("fused MoE75 invalid scratch geometry".into());
    }
    Ok(Arc::new(Plan {
        workspace_bytes,
        map_bytes,
        route_scratch_bytes: tokens * 8 * 8,
        selected_gemm1: first,
        selected_gemm2: second,
        tokens,
        packed,
        runner: Mutex::new(runner),
    }))
}

fn validate_ranges(ranges: &[(u64, usize)]) -> Result<(), String> {
    for (i, &(start, length)) in ranges.iter().enumerate() {
        if start == 0 || length == 0 {
            return Err("fused MoE75 null/empty buffer".into());
        }
        let end = start
            .checked_add(u64::try_from(length).map_err(|_| "fused MoE75 length overflow")?)
            .ok_or("fused MoE75 address overflow")?;
        for &(other, bytes) in &ranges[..i] {
            let other_end = other
                .checked_add(bytes as u64)
                .ok_or("fused MoE75 address overflow")?;
            if start < other_end && other < end {
                return Err("fused MoE75 overlapping buffer ranges".into());
            }
        }
    }
    Ok(())
}
impl Plan {
    /// A private owned byte view for lowered persistent-resource accounting.
    pub(crate) fn packed_buffer(&self) -> CudaBuffer<u8> {
        self.packed.buffer.clone()
    }
    /// Submit on the exact owning primary stream. Caller holds the device execution
    /// gate through completion and retains this plan and all buffers until a checked
    /// invocation fence, including errors/cancellation. On a failed fence it must
    /// quarantine these resources. A device preflight sanitizes invalid route IDs;
    /// all buffers must have the capacities derived below and belong to this context.
    /// Whole-read71 may capture this submission on the same primary stream;
    /// its private frame must retain this plan (including runner and packed
    /// weights), all referenced buffers, and the device gate through completion.
    pub(crate) unsafe fn launch(
        &self,
        stream: &Arc<CudaStream>,
        addresses: [u64; 7],
        status: u64,
        error_context: u32,
        route_scratch: u64,
    ) -> Result<(), String> {
        let runner = self
            .runner
            .lock()
            .map_err(|_| "fused MoE75 runner lock poisoned")?;
        if !Arc::ptr_eq(stream, &runner.stream) {
            return Err("fused MoE75 stream ownership differs".into());
        }
        stream
            .context()
            .bind_to_thread()
            .map_err(|e| e.to_string())?;
        let [input, experts, scales, down, output, workspace, map] = addresses;
        let ranges = [
            (input, self.tokens * 2816 * 2),
            (experts, self.tokens * 8 * 4),
            (scales, self.tokens * 8 * 4),
            (down, DOWN_BYTES),
            (output, self.tokens * 2816 * 2),
            (workspace, self.workspace_bytes),
            (map, self.map_bytes),
            (self.packed.buffer.address(), GATE_BYTES),
            (status, 8),
            (route_scratch, self.route_scratch_bytes),
        ];
        validate_ranges(&ranges)?;
        if input % 16 != 0
            || experts % 4 != 0
            || scales % 4 != 0
            || down % 16 != 0
            || output % 16 != 0
            || workspace % 256 != 0
            || map % 16 != 0
            || status % 8 != 0
            || route_scratch % 4 != 0
        {
            return Err("fused MoE75 buffer alignment differs".into());
        }
        let sanitized_scales = route_scratch + (self.tokens * 8 * 4) as u64;
        let tokens = self.tokens as u32;
        let mut guard = stream.launch_builder(&self.packed.guard.function);
        guard
            .arg(&experts)
            .arg(&scales)
            .arg(&route_scratch)
            .arg(&sanitized_scales)
            .arg(&status)
            .arg(&error_context)
            .arg(&tokens);
        unsafe {
            guard.launch(LaunchConfig {
                grid_dim: (tokens.div_ceil(128), 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .map_err(|e| format!("fused MoE75 route guard: {e}"))?;
        let mut error = [0; ERROR_BYTES];
        let result = unsafe {
            (runner.api.run)(
                runner.handle,
                self.tokens as u32,
                input as *const c_void,
                route_scratch as *const i32,
                sanitized_scales as *const f32,
                self.packed.buffer.address() as *const c_void,
                down as *const c_void,
                output as *mut c_void,
                workspace as *mut c_void,
                self.workspace_bytes,
                map as *mut c_void,
                self.map_bytes,
                stream.cu_stream().cast(),
                error.as_mut_ptr(),
                ERROR_BYTES,
            )
        };
        checked(result, &error, "run")
    }
}

#[cfg(test)]
mod tests {
    use super::{
        checked, parse_tactic, proves_route_ranks, same_source, validate_ranges, ERROR_BYTES,
    };
    use effect_torch_compiler::{DenseNodeId, ExpertRouteRankRegion, NativeRegion};
    use std::sync::Arc;
    #[test]
    fn native_finalizer_requires_proved_ranks_of_the_same_topk() {
        let id = |index| DenseNodeId::from_index(index).unwrap();
        let ranks = id(2);
        let experts = id(0);
        let mut proof = ExpertRouteRankRegion {
            nodes: vec![ranks].into_boxed_slice(),
            inputs: vec![experts, id(1)].into_boxed_slice(),
            output: ranks,
            rows: 64,
            routes: 8,
        };
        let admitted = |proof: &ExpertRouteRankRegion, output, topk, tokens| {
            proves_route_ranks(
                Some(&NativeRegion::ExpertRouteRank(proof.clone())),
                output,
                topk,
                tokens,
            )
        };
        assert!(admitted(&proof, ranks, experts, 64));
        assert!(!proves_route_ranks(None, ranks, experts, 64));
        assert!(!admitted(&proof, id(3), experts, 64));
        assert!(!admitted(&proof, ranks, id(3), 64));
        assert!(!admitted(&proof, ranks, experts, 256));
        proof.routes = 7;
        assert!(!admitted(&proof, ranks, experts, 64));
        proof.routes = 8;
        proof.inputs = Box::new([]);
        assert!(!admitted(&proof, ranks, experts, 64));
    }
    #[test]
    fn actual_route_guard_preserves_sticky_errors_and_sanitizes_before_ffi() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let binary =
            std::env::temp_dir().join(format!("moe75-guard-{}-{unique}", std::process::id()));
        let source = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("src/kernels/fused_moe75_guard_host.cpp");
        let compiler = std::env::var_os("CXX").unwrap_or_else(|| "c++".into());
        let output = std::process::Command::new(compiler)
            .args(["-std=c++17", "-O2"])
            .arg(source)
            .arg("-o")
            .arg(&binary)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let output = std::process::Command::new(&binary).output().unwrap();
        std::fs::remove_file(binary).unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
    }
    #[test]
    fn tactic_defaults_and_invalid_values() {
        assert_eq!(parse_tactic(None).unwrap(), -1);
        assert_eq!(parse_tactic(Some("-1")).unwrap(), -1);
        assert_eq!(parse_tactic(Some("0")).unwrap(), 0);
        assert_eq!(parse_tactic(Some("31")).unwrap(), 31);
        for invalid in ["-2", "", "1.2", "2147483648"] {
            assert!(parse_tactic(Some(invalid)).is_err());
        }
    }
    #[test]
    fn cache_identity_requires_live_exact_owner() {
        let first = Arc::new(7);
        let other = Arc::new(7);
        let witness = Arc::downgrade(&first);
        assert!(same_source(&witness, &Arc::clone(&first)));
        assert!(!same_source(&witness, &other));
        drop(first);
        assert!(!same_source(&witness, &other));
    }
    #[test]
    fn foreign_error_message_is_bounded_even_without_terminator() {
        let error = [b'x' as std::ffi::c_char; ERROR_BYTES];
        assert!(checked(0, &error, "test").is_ok());
        let message = checked(1, &error, "test").unwrap_err();
        assert!(message.ends_with(&"x".repeat(ERROR_BYTES)));
    }
    #[test]
    fn ranges_reject_partial_aliases_and_overflow() {
        assert!(validate_ranges(&[(100, 20), (119, 20)]).is_err());
        assert!(validate_ranges(&[(100, 20), (120, 20)]).is_ok());
        assert!(validate_ranges(&[(u64::MAX - 2, 4)]).is_err());
        assert!(validate_ranges(&[(0, 4)]).is_err());
        assert!(validate_ranges(&[(100, 0)]).is_err());
    }
}
