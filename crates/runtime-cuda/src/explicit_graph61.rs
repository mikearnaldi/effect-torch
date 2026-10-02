//! Owned explicit kernel DAGs for the narrow expert-pair experiment.
//!
//! No stream capture occurs here. Graph nodes copy their argument values during
//! construction; callers own the pointed-to device allocations through execution.
use crate::executable::CudaKernelArgs;
use cudarc::driver::{sys, CudaContext, CudaStream, LaunchConfig};
use sha2::{Digest, Sha256};
use std::ffi::{c_void, CString};
use std::ptr;
use std::sync::{Arc, Mutex};

trait Driver61: Send + Sync {
    fn bind(&self) -> Result<(), String>;
    fn context_id(&self) -> usize;
    fn load_module(&self, bytes: &CString) -> Result<sys::CUmodule, String>;
    fn function(&self, module: sys::CUmodule, name: &CString) -> Result<sys::CUfunction, String>;
    fn unload_module(&self, module: sys::CUmodule) -> Result<(), String>;
    fn create_graph(&self) -> Result<sys::CUgraph, String>;
    fn add_kernel(
        &self,
        graph: sys::CUgraph,
        parents: &[sys::CUgraphNode],
        params: &sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<sys::CUgraphNode, String>;
    fn instantiate(&self, graph: sys::CUgraph) -> Result<sys::CUgraphExec, String>;
    fn destroy_exec(&self, exec: sys::CUgraphExec) -> Result<(), String>;
    fn destroy_graph(&self, graph: sys::CUgraph) -> Result<(), String>;
    fn launch(&self, exec: sys::CUgraphExec, stream: sys::CUstream) -> Result<(), String>;
}
struct CudaDriver61 {
    context: Arc<CudaContext>,
}
fn checked(result: sys::CUresult, operation: &str) -> Result<(), String> {
    result
        .result()
        .map_err(|error| format!("explicit graph61 {operation}: {error}"))
}
impl Driver61 for CudaDriver61 {
    fn bind(&self) -> Result<(), String> {
        self.context
            .bind_to_thread()
            .map_err(|e| format!("explicit graph61 bind: {e}"))
    }
    fn context_id(&self) -> usize {
        self.context.cu_ctx() as usize
    }
    fn load_module(&self, bytes: &CString) -> Result<sys::CUmodule, String> {
        let mut module = ptr::null_mut();
        checked(
            unsafe { sys::cuModuleLoadData(&mut module, bytes.as_ptr().cast()) },
            "load module",
        )?;
        Ok(module)
    }
    fn function(&self, module: sys::CUmodule, name: &CString) -> Result<sys::CUfunction, String> {
        let mut function = ptr::null_mut();
        checked(
            unsafe { sys::cuModuleGetFunction(&mut function, module, name.as_ptr()) },
            "module function",
        )?;
        Ok(function)
    }
    fn unload_module(&self, module: sys::CUmodule) -> Result<(), String> {
        checked(unsafe { sys::cuModuleUnload(module) }, "unload module")
    }
    fn create_graph(&self) -> Result<sys::CUgraph, String> {
        let mut graph = ptr::null_mut();
        checked(unsafe { sys::cuGraphCreate(&mut graph, 0) }, "create")?;
        Ok(graph)
    }
    fn add_kernel(
        &self,
        graph: sys::CUgraph,
        parents: &[sys::CUgraphNode],
        params: &sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<sys::CUgraphNode, String> {
        let mut node = ptr::null_mut();
        let dependencies = if parents.is_empty() {
            ptr::null()
        } else {
            parents.as_ptr()
        };
        checked(
            unsafe {
                sys::cuGraphAddKernelNode_v2(&mut node, graph, dependencies, parents.len(), params)
            },
            "add kernel v2",
        )?;
        Ok(node)
    }
    fn instantiate(&self, graph: sys::CUgraph) -> Result<sys::CUgraphExec, String> {
        let mut exec = ptr::null_mut();
        checked(
            unsafe { sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
            "instantiate",
        )?;
        Ok(exec)
    }
    fn destroy_exec(&self, exec: sys::CUgraphExec) -> Result<(), String> {
        checked(unsafe { sys::cuGraphExecDestroy(exec) }, "destroy exec")
    }
    fn destroy_graph(&self, graph: sys::CUgraph) -> Result<(), String> {
        checked(unsafe { sys::cuGraphDestroy(graph) }, "destroy graph")
    }
    fn launch(&self, exec: sys::CUgraphExec, stream: sys::CUstream) -> Result<(), String> {
        checked(unsafe { sys::cuGraphLaunch(exec, stream) }, "launch")
    }
}

pub(crate) struct RawModule61 {
    driver: Arc<dyn Driver61>,
    raw: sys::CUmodule,
    artifact: [u8; 32],
}
// CUDA modules/functions are immutable after loading. Every driver operation
// binds the owning context; Arc prevents unloading during lookup or graph use.
unsafe impl Send for RawModule61 {}
unsafe impl Sync for RawModule61 {}
impl RawModule61 {
    pub(crate) fn load(context: Arc<CudaContext>, bytes: &[u8]) -> Result<Arc<Self>, String> {
        Self::load_with(Arc::new(CudaDriver61 { context }), bytes)
    }
    fn load_with(driver: Arc<dyn Driver61>, bytes: &[u8]) -> Result<Arc<Self>, String> {
        if bytes.is_empty() {
            return Err("explicit graph61 empty PTX".into());
        }
        let cbytes = if bytes.last() == Some(&0) {
            CString::from_vec_with_nul(bytes.to_vec())
        } else {
            CString::from_vec_with_nul(bytes.iter().copied().chain([0]).collect())
        }
        .map_err(|_| "explicit graph61 PTX has interior NUL")?;
        driver.bind()?;
        let raw = driver.load_module(&cbytes)?;
        if raw.is_null() {
            return Err("explicit graph61 module returned null".into());
        }
        Ok(Arc::new(Self {
            driver,
            raw,
            artifact: Sha256::digest(bytes).into(),
        }))
    }
    pub(crate) fn function(self: &Arc<Self>, symbol: &str) -> Result<Arc<RawKernel61>, String> {
        let name = CString::new(symbol).map_err(|_| "explicit graph61 symbol has NUL")?;
        if symbol.is_empty() {
            return Err("explicit graph61 empty symbol".into());
        }
        self.driver.bind()?;
        let raw = self.driver.function(self.raw, &name)?;
        if raw.is_null() {
            return Err("explicit graph61 function returned null".into());
        }
        Ok(Arc::new(RawKernel61 {
            module: self.clone(),
            raw,
            symbol: symbol.into(),
        }))
    }
}
impl Drop for RawModule61 {
    fn drop(&mut self) {
        if let Err(error) = self
            .driver
            .bind()
            .and_then(|()| self.driver.unload_module(self.raw))
        {
            // Never unload under an unbound context or drop its final owner when
            // the driver cannot prove release. A destructor must not panic.
            eprintln!("{error}; retaining explicit graph61 module context");
            std::mem::forget(self.driver.clone());
        }
    }
}
pub(crate) struct RawKernel61 {
    module: Arc<RawModule61>,
    raw: sys::CUfunction,
    symbol: String,
}
impl std::fmt::Debug for RawKernel61 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawKernel61")
            .field("symbol", &self.symbol)
            .field("artifact", &self.module.artifact)
            .finish()
    }
}
// SAFETY: function handles are immutable and retain their context-bound module.
unsafe impl Send for RawKernel61 {}
unsafe impl Sync for RawKernel61 {}

#[derive(Clone, Debug)]
pub(crate) enum NodeArgs61 {
    Typed(CudaKernelArgs),
    Merged {
        descriptors: u64,
        shapes: u64,
        count: u32,
        columns: u32,
        inner: u32,
    },
    M1 {
        descriptors: u64,
        m1: u64,
    },
}
impl NodeArgs61 {
    // Host storage stays live throughout cuGraphAddKernelNode_v2, which copies
    // values according to the function ABI. No struct padding is serialized.
    fn pointers(&mut self) -> Vec<*mut c_void> {
        match self {
            Self::Typed(a) => vec![(a as *mut CudaKernelArgs).cast()],
            Self::Merged {
                descriptors,
                shapes,
                count,
                columns,
                inner,
            } => vec![
                (descriptors as *mut u64).cast(),
                (shapes as *mut u64).cast(),
                (count as *mut u32).cast(),
                (columns as *mut u32).cast(),
                (inner as *mut u32).cast(),
            ],
            Self::M1 { descriptors, m1 } => {
                vec![(descriptors as *mut u64).cast(), (m1 as *mut u64).cast()]
            }
        }
    }
    pub(crate) fn key(&self) -> Vec<u8> {
        let mut out = Vec::new();
        let put64 = |out: &mut Vec<u8>, v: u64| out.extend(v.to_le_bytes());
        let put32 = |out: &mut Vec<u8>, v: u32| out.extend(v.to_le_bytes());
        match self {
            Self::Typed(a) => {
                out.push(0);
                for v in a.inputs {
                    put64(&mut out, v)
                }
                put64(&mut out, a.output);
                for v in a.scratch {
                    put64(&mut out, v)
                }
                put64(&mut out, a.metadata);
                put64(&mut out, a.elements);
                for v in a.integers {
                    put64(&mut out, v)
                }
                for v in a.scalars {
                    put64(&mut out, v.to_bits())
                }
                for v in a.input_dtypes {
                    put32(&mut out, v)
                }
                for v in [
                    a.output_dtype,
                    a.compute_dtype,
                    a.operation,
                    a.error_context,
                ] {
                    put32(&mut out, v)
                }
            }
            Self::Merged {
                descriptors,
                shapes,
                count,
                columns,
                inner,
            } => {
                out.push(1);
                put64(&mut out, *descriptors);
                put64(&mut out, *shapes);
                for v in [count, columns, inner] {
                    put32(&mut out, *v)
                }
            }
            Self::M1 { descriptors, m1 } => {
                out.push(2);
                put64(&mut out, *descriptors);
                put64(&mut out, *m1)
            }
        }
        out
    }
}
#[derive(Clone)]
pub(crate) struct KernelNodeSpec61 {
    pub(crate) kernel: Arc<RawKernel61>,
    pub(crate) config: LaunchConfig,
    pub(crate) args: NodeArgs61,
    pub(crate) parents: Vec<usize>,
}
impl KernelNodeSpec61 {
    pub(crate) fn key_bytes(&self) -> Vec<u8> {
        self.key()
    }
    /// Complete function, context, launch, argument and dependency contribution.
    pub(crate) fn key(&self) -> Vec<u8> {
        let mut key = self.kernel.module.artifact.to_vec();
        key.extend((self.kernel.module.driver.context_id() as u64).to_le_bytes());
        key.extend((self.kernel.symbol.len() as u64).to_le_bytes());
        key.extend(self.kernel.symbol.as_bytes());
        for v in [
            self.config.grid_dim.0,
            self.config.grid_dim.1,
            self.config.grid_dim.2,
            self.config.block_dim.0,
            self.config.block_dim.1,
            self.config.block_dim.2,
            self.config.shared_mem_bytes,
        ] {
            key.extend(v.to_le_bytes())
        }
        key.extend((self.parents.len() as u64).to_le_bytes());
        for &p in &self.parents {
            key.extend((p as u64).to_le_bytes())
        }
        key.extend(self.args.key());
        key
    }
}

struct OwnedGraph61 {
    driver: Arc<dyn Driver61>,
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    kernels: Vec<Arc<RawKernel61>>,
}
// SAFETY: construction mutates handles before publication. Published graphs
// are immutable, launch calls are serialized, and Drop has exclusive ownership.
unsafe impl Send for OwnedGraph61 {}
unsafe impl Sync for OwnedGraph61 {}
impl Drop for OwnedGraph61 {
    fn drop(&mut self) {
        let cleanup = self.driver.bind().and_then(|()| {
            if !self.exec.is_null() {
                self.driver.destroy_exec(self.exec)?;
                self.exec = ptr::null_mut();
            }
            if !self.graph.is_null() {
                self.driver.destroy_graph(self.graph)?;
                self.graph = ptr::null_mut();
            }
            Ok(())
        });
        if let Err(error) = cleanup {
            // A live raw graph may still reference its modules. On failed
            // destruction leak the dependency owners together, never just them.
            eprintln!("{error}; retaining explicit graph61 dependencies");
            std::mem::forget(std::mem::take(&mut self.kernels));
            std::mem::forget(self.driver.clone());
        }
    }
}
pub(crate) struct ExplicitGraph61 {
    owned: OwnedGraph61,
    submission: Mutex<()>,
}
impl ExplicitGraph61 {
    /// # Safety
    /// Each NodeArgs61 variant must exactly match the registered function ABI;
    /// CUDA copies argument-sized host values during node construction.
    pub(crate) unsafe fn build(
        context: Arc<CudaContext>,
        nodes: &[KernelNodeSpec61],
    ) -> Result<Arc<Self>, String> {
        unsafe { Self::build_with(Arc::new(CudaDriver61 { context }), nodes) }
    }
    unsafe fn build_with(
        driver: Arc<dyn Driver61>,
        nodes: &[KernelNodeSpec61],
    ) -> Result<Arc<Self>, String> {
        if nodes.is_empty() {
            return Err("explicit graph61 empty DAG".into());
        }
        // Complete pure preflight before any driver/resource mutation.
        for (i, node) in nodes.iter().enumerate() {
            if node.kernel.module.driver.context_id() != driver.context_id() {
                return Err("explicit graph61 kernel context mismatch".into());
            }
            if [
                node.config.grid_dim.0,
                node.config.grid_dim.1,
                node.config.grid_dim.2,
                node.config.block_dim.0,
                node.config.block_dim.1,
                node.config.block_dim.2,
            ]
            .contains(&0)
            {
                return Err("explicit graph61 zero launch extent".into());
            }
            if node.parents.iter().any(|&p| p >= i)
                || node
                    .parents
                    .iter()
                    .enumerate()
                    .any(|(j, p)| node.parents[..j].contains(p))
            {
                return Err("explicit graph61 invalid/duplicate dependency".into());
            }
        }
        driver.bind()?;
        let graph = driver.create_graph()?;
        if graph.is_null() {
            return Err("explicit graph61 graph returned null".into());
        }
        let mut owned = OwnedGraph61 {
            driver: driver.clone(),
            graph,
            exec: ptr::null_mut(),
            kernels: nodes.iter().map(|n| n.kernel.clone()).collect(),
        };
        let mut handles = Vec::with_capacity(nodes.len());
        for node in nodes {
            let parents = node.parents.iter().map(|&p| handles[p]).collect::<Vec<_>>();
            let mut args = node.args.clone();
            let mut pointers = args.pointers();
            let params = sys::CUDA_KERNEL_NODE_PARAMS {
                func: node.kernel.raw,
                gridDimX: node.config.grid_dim.0,
                gridDimY: node.config.grid_dim.1,
                gridDimZ: node.config.grid_dim.2,
                blockDimX: node.config.block_dim.0,
                blockDimY: node.config.block_dim.1,
                blockDimZ: node.config.block_dim.2,
                sharedMemBytes: node.config.shared_mem_bytes,
                kernelParams: pointers.as_mut_ptr(),
                extra: ptr::null_mut(),
                kern: ptr::null_mut(),
                ctx: ptr::null_mut(),
            };
            let handle = driver.add_kernel(graph, &parents, &params)?;
            if handle.is_null() {
                return Err("explicit graph61 node returned null".into());
            }
            handles.push(handle);
        }
        owned.exec = driver.instantiate(graph)?;
        if owned.exec.is_null() {
            return Err("explicit graph61 executable returned null".into());
        }
        Ok(Arc::new(Self {
            owned,
            submission: Mutex::new(()),
        }))
    }
    /// # Safety
    /// Every raw device address must satisfy its kernel's ABI, extents and alias
    /// rules. Keep this graph and all referenced allocations alive until the
    /// stream has completed, including after a launch error. Never replay eager
    /// work after a submission error without proving it was not submitted.
    /// This method neither captures nor synchronizes.
    pub(crate) unsafe fn launch(&self, stream: &Arc<CudaStream>) -> Result<(), String> {
        if stream.context().cu_ctx() as usize != self.owned.driver.context_id() {
            return Err("explicit graph61 stream context mismatch".into());
        }
        let _submission = self
            .submission
            .lock()
            .map_err(|_| "explicit graph61 submission lock poisoned")?;
        self.owned.driver.bind()?;
        self.owned
            .driver
            .launch(self.owned.exec, stream.cu_stream())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[derive(Default)]
    struct State {
        events: Vec<String>,
        fail: Option<String>,
        also_fail: Option<String>,
        adds: usize,
        copied: Vec<Vec<u8>>,
    }
    #[derive(Default)]
    struct Mock {
        state: Mutex<State>,
        identity: usize,
    }
    impl Mock {
        fn event(&self, name: &str) -> Result<(), String> {
            let mut s = self.state.lock().unwrap();
            s.events.push(name.into());
            if s.fail.as_deref() == Some(name) || s.also_fail.as_deref() == Some(name) {
                Err(format!("injected {name}"))
            } else {
                Ok(())
            }
        }
        fn events(&self) -> Vec<String> {
            self.state.lock().unwrap().events.clone()
        }
        fn fail(&self, name: &str) {
            self.state.lock().unwrap().fail = Some(name.into());
        }
    }
    impl Driver61 for Mock {
        fn bind(&self) -> Result<(), String> {
            self.event("bind")
        }
        fn context_id(&self) -> usize {
            self.identity
        }
        fn load_module(&self, _: &CString) -> Result<sys::CUmodule, String> {
            self.event("load")?;
            Ok(1usize as _)
        }
        fn function(&self, _: sys::CUmodule, name: &CString) -> Result<sys::CUfunction, String> {
            self.event("function")?;
            Ok(match name.to_str().unwrap() {
                "typed" => 11usize,
                "merged" => 12,
                "m1" => 13,
                _ => 14,
            } as _)
        }
        fn unload_module(&self, _: sys::CUmodule) -> Result<(), String> {
            self.event("unload")
        }
        fn create_graph(&self) -> Result<sys::CUgraph, String> {
            self.event("create")?;
            Ok(2usize as _)
        }
        fn add_kernel(
            &self,
            _: sys::CUgraph,
            parents: &[sys::CUgraphNode],
            p: &sys::CUDA_KERNEL_NODE_PARAMS,
        ) -> Result<sys::CUgraphNode, String> {
            let n = {
                let mut s = self.state.lock().unwrap();
                let n = s.adds;
                s.adds += 1;
                n
            };
            self.event(&format!("add{n}"))?;
            assert!(p.kern.is_null() && p.ctx.is_null() && p.extra.is_null());
            for &parent in parents {
                assert!((parent as usize) >= 100 && (parent as usize) < 100 + n);
            }
            let copied = unsafe {
                match p.func as usize {
                    11 => {
                        NodeArgs61::Typed(*(p.kernelParams.read() as *const CudaKernelArgs)).key()
                    }
                    12 => NodeArgs61::Merged {
                        descriptors: *(p.kernelParams.read() as *const u64),
                        shapes: *(p.kernelParams.add(1).read() as *const u64),
                        count: *(p.kernelParams.add(2).read() as *const u32),
                        columns: *(p.kernelParams.add(3).read() as *const u32),
                        inner: *(p.kernelParams.add(4).read() as *const u32),
                    }
                    .key(),
                    13 => NodeArgs61::M1 {
                        descriptors: *(p.kernelParams.read() as *const u64),
                        m1: *(p.kernelParams.add(1).read() as *const u64),
                    }
                    .key(),
                    _ => panic!("unexpected ABI"),
                }
            };
            self.state.lock().unwrap().copied.push(copied);
            Ok((100 + n) as _)
        }
        fn instantiate(&self, _: sys::CUgraph) -> Result<sys::CUgraphExec, String> {
            self.event("instantiate")?;
            Ok(3usize as _)
        }
        fn destroy_exec(&self, _: sys::CUgraphExec) -> Result<(), String> {
            self.event("destroy_exec")
        }
        fn destroy_graph(&self, _: sys::CUgraph) -> Result<(), String> {
            self.event("destroy_graph")
        }
        fn launch(&self, _: sys::CUgraphExec, _: sys::CUstream) -> Result<(), String> {
            self.event("launch")
        }
    }
    fn nodes(driver: &Arc<Mock>) -> Vec<KernelNodeSpec61> {
        let module = RawModule61::load_with(driver.clone(), b"literal PTX bytes").unwrap();
        let config = LaunchConfig {
            grid_dim: (64, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 17,
        };
        let mut typed = CudaKernelArgs::default();
        typed.inputs[7] = 77;
        typed.output = 99;
        typed.scalars[3] = f64::from_bits(0xfff8000000000042);
        typed.error_context = 0xabcdef;
        vec![
            KernelNodeSpec61 {
                kernel: module.function("typed").unwrap(),
                config,
                args: NodeArgs61::Typed(typed),
                parents: vec![],
            },
            KernelNodeSpec61 {
                kernel: module.function("merged").unwrap(),
                config,
                args: NodeArgs61::Merged {
                    descriptors: 1,
                    shapes: 2,
                    count: 3,
                    columns: 4,
                    inner: 5,
                },
                parents: vec![0],
            },
            KernelNodeSpec61 {
                kernel: module.function("m1").unwrap(),
                config,
                args: NodeArgs61::M1 {
                    descriptors: 6,
                    m1: 7,
                },
                parents: vec![0, 1],
            },
        ]
    }
    #[test]
    fn explicit61_arguments_copied_and_modules_outlive_graph() {
        let driver = Arc::new(Mock::default());
        let nodes = nodes(&driver);
        let module = Arc::downgrade(&nodes[0].kernel.module);
        let graph = unsafe { ExplicitGraph61::build_with(driver.clone(), &nodes) }.unwrap();
        assert_eq!(
            driver.state.lock().unwrap().copied,
            nodes.iter().map(|n| n.args.key()).collect::<Vec<_>>()
        );
        drop(nodes);
        assert!(module.upgrade().is_some());
        assert!(!driver.events().contains(&"unload".into()));
        drop(graph);
        assert!(module.upgrade().is_none());
        let events = driver.events();
        let locate = |name: &str| events.iter().position(|v| v == name).unwrap();
        assert!(
            locate("destroy_exec") < locate("destroy_graph")
                && locate("destroy_graph") < locate("unload")
        );
    }
    fn expert_pair_nodes(driver: &Arc<Mock>) -> Vec<KernelNodeSpec61> {
        let templates = nodes(driver);
        (0..15)
            .map(|i| {
                let mut n = templates[i % templates.len()].clone();
                n.parents = match i {
                    0 => vec![],
                    1..=6 => vec![i - 1],
                    7 | 8 => vec![6],
                    9 => vec![7, 8],
                    10 | 11 => vec![i - 1],
                    12 | 13 => vec![11],
                    14 => vec![12, 13],
                    _ => unreachable!(),
                };
                n
            })
            .collect()
    }
    #[test]
    fn explicit61_partial_build_failures_release_owned_graph_once() {
        let failures = std::iter::once("create".to_string())
            .chain((0..15).map(|i| format!("add{i}")))
            .chain(["instantiate".to_string()]);
        for failure in failures {
            let driver = Arc::new(Mock::default());
            let nodes = expert_pair_nodes(&driver);
            assert_eq!(nodes.iter().map(|n| n.parents.len()).sum::<usize>(), 16);
            driver.fail(&failure);
            let err = unsafe { ExplicitGraph61::build_with(driver.clone(), &nodes) }
                .err()
                .unwrap();
            assert_eq!(err, format!("injected {failure}"));
            let events = driver.events();
            assert_eq!(
                events
                    .iter()
                    .filter(|v| v.as_str() == "destroy_graph")
                    .count(),
                usize::from(failure != "create")
            );
            assert!(!events.contains(&"destroy_exec".into()));
            drop(nodes);
            assert_eq!(
                driver
                    .events()
                    .iter()
                    .filter(|v| v.as_str() == "unload")
                    .count(),
                1
            );
        }
    }
    #[test]
    fn explicit61_original_build_error_survives_cleanup_failure() {
        let driver = Arc::new(Mock::default());
        let nodes = expert_pair_nodes(&driver);
        driver.fail("add8");
        driver.state.lock().unwrap().also_fail = Some("destroy_graph".into());
        let module = Arc::downgrade(&nodes[0].kernel.module);
        let err = unsafe { ExplicitGraph61::build_with(driver.clone(), &nodes) }
            .err()
            .unwrap();
        assert_eq!(err, "injected add8");
        drop(nodes);
        assert!(module.upgrade().is_some());
        assert!(!driver.events().contains(&"unload".into()));
    }
    #[test]
    fn explicit61_preflight_rejects_context_dependencies_and_geometry_without_driver_work() {
        for case in 0..5 {
            let driver = Arc::new(Mock::default());
            let mut nodes = nodes(&driver);
            match case {
                0 => nodes[0].parents = vec![0],
                1 => nodes[2].parents = vec![0, 0],
                2 => nodes[0].config.grid_dim.0 = 0,
                3 => nodes.clear(),
                _ => {}
            }
            let before = driver.events();
            let api: Arc<dyn Driver61> = if case == 4 {
                Arc::new(Mock {
                    identity: 1,
                    ..Default::default()
                })
            } else {
                driver.clone()
            };
            assert!(unsafe { ExplicitGraph61::build_with(api, &nodes) }.is_err());
            assert_eq!(driver.events(), before);
        }
    }
    #[test]
    fn explicit61_keys_cover_arguments_geometry_symbol_artifact_and_dependencies() {
        let driver = Arc::new(Mock::default());
        let nodes = nodes(&driver);
        let original = nodes[0].key_bytes();
        for case in 0..6 {
            let mut n = nodes[0].clone();
            match case {
                0 => n.config.grid_dim.0 += 1,
                1 => n.config.shared_mem_bytes += 1,
                2 => n.parents.push(0),
                3 => n.kernel = nodes[1].kernel.clone(),
                4 => {
                    let NodeArgs61::Typed(ref mut a) = n.args else {
                        unreachable!()
                    };
                    a.scalars[3] = f64::from_bits(a.scalars[3].to_bits() ^ 1)
                }
                _ => {
                    let NodeArgs61::Typed(ref mut a) = n.args else {
                        unreachable!()
                    };
                    a.error_context ^= 1;
                }
            }
            assert_ne!(original, n.key_bytes());
        }
        let other = RawModule61::load_with(driver, b"different PTX bytes").unwrap();
        let mut n = nodes[0].clone();
        n.kernel = other.function("typed").unwrap();
        assert_ne!(original, n.key_bytes());
        assert_eq!(nodes[0].args.key().len(), 361);
    }
    #[test]
    fn explicit61_module_errors_cleanup_and_no_panic_destructors() {
        let driver = Arc::new(Mock::default());
        assert!(RawModule61::load_with(driver.clone(), b"bad\0interior").is_err());
        assert!(driver.events().is_empty());
        driver.fail("load");
        assert!(RawModule61::load_with(driver.clone(), b"ptx").is_err());
        assert!(!driver.events().contains(&"unload".into()));
        driver.state.lock().unwrap().fail = None;
        let module = RawModule61::load_with(driver.clone(), b"ptx").unwrap();
        driver.fail("function");
        assert!(module.function("typed").is_err());
        drop(module);
        assert_eq!(
            driver
                .events()
                .iter()
                .filter(|v| v.as_str() == "unload")
                .count(),
            1
        );
    }
    #[test]
    fn explicit61_failed_destruction_retains_module_dependencies() {
        for failure in ["bind", "destroy_exec", "destroy_graph"] {
            let driver = Arc::new(Mock::default());
            let nodes = nodes(&driver);
            let module = Arc::downgrade(&nodes[0].kernel.module);
            let graph = unsafe { ExplicitGraph61::build_with(driver.clone(), &nodes) }.unwrap();
            drop(nodes);
            driver.fail(failure);
            drop(graph);
            assert!(module.upgrade().is_some());
            assert!(!driver.events().contains(&"unload".into()));
            // Fault-injected retained owners intentionally model fail-closed
            // shutdown; tests never dereference these fake CUDA handles.
        }
    }
}

#[cfg(test)]
mod hardware_tests {
    use super::*;
    use cudarc::driver::DevicePtr;
    const ADD_ONE: &[u8] = br#"
.version 8.0
.target sm_80
.address_size 64
.visible .entry add_one(.param .u64 output, .param .u64 input) {
    .reg .u64 %rd<2>;
    .reg .u32 %r<2>;
    ld.param.u64 %rd0, [output];
    ld.param.u64 %rd1, [input];
    ld.global.u32 %r0, [%rd1];
    add.u32 %r1, %r0, 1;
    st.global.u32 [%rd0], %r1;
    ret;
}
"#;
    #[test]
    #[ignore = "requires CUDA: explicit raw PTX DAG launch, failed-build cleanup, contexts and retention"]
    fn explicit61_raw_ptx_dag_launch_context_failures_and_module_retention() {
        let context = CudaContext::new(0).unwrap();
        let stream = context.new_stream().unwrap();
        assert!(RawModule61::load(context.clone(), b"invalid PTX").is_err());
        let module = RawModule61::load(context.clone(), ADD_ONE).unwrap();
        assert!(module.function("missing_entrypoint").is_err());
        let weak = Arc::downgrade(&module);
        let kernel = module.function("add_one").unwrap();
        let input = stream.clone_htod(&[40u32]).unwrap();
        let middle = stream.alloc_zeros::<u32>(1).unwrap();
        let output = stream.alloc_zeros::<u32>(1).unwrap();
        let address = |value: &cudarc::driver::CudaSlice<u32>| {
            let (ptr, guard) = value.device_ptr(&stream);
            drop(guard);
            ptr
        };
        let config = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1, 1, 1),
            shared_mem_bytes: 0,
        };
        let nodes = vec![
            KernelNodeSpec61 {
                kernel: kernel.clone(),
                config,
                args: NodeArgs61::M1 {
                    descriptors: address(&middle),
                    m1: address(&input),
                },
                parents: vec![],
            },
            KernelNodeSpec61 {
                kernel: kernel.clone(),
                config,
                args: NodeArgs61::M1 {
                    descriptors: address(&output),
                    m1: address(&middle),
                },
                parents: vec![0],
            },
        ];
        stream.synchronize().unwrap();
        // Exact two-u64 ABI matches this test PTX. The third invalid launch
        // requests a real CUDA constructor error after two admitted nodes.
        let mut invalid = nodes.clone();
        let mut last = nodes[1].clone();
        last.config.block_dim = (4096, 1, 1);
        last.parents = vec![1];
        invalid.push(last);
        assert!(unsafe { ExplicitGraph61::build(context.clone(), &invalid) }.is_err());
        drop(invalid);
        let graph = unsafe { ExplicitGraph61::build(context.clone(), &nodes) }.unwrap();
        let other = CudaContext::new_non_primary(0, 0).unwrap();
        let other_stream = other.new_stream().unwrap();
        assert!(unsafe { ExplicitGraph61::build(other.clone(), &nodes) }.is_err());
        assert!(unsafe { graph.launch(&other_stream) }.is_err());
        drop(other_stream);
        drop(other);
        drop(nodes);
        drop(kernel);
        drop(module);
        assert!(weak.upgrade().is_some());
        // Hold graph and all three allocations through completion. Repeated
        // graph launches reuse addresses but never overwrite the input.
        for _ in 0..3 {
            unsafe { graph.launch(&stream) }.unwrap();
            stream.synchronize().unwrap();
            assert_eq!(stream.clone_dtoh(&output).unwrap(), vec![42]);
            assert_eq!(stream.clone_dtoh(&input).unwrap(), vec![40]);
        }
        drop(graph);
        assert!(weak.upgrade().is_none());
    }
}
