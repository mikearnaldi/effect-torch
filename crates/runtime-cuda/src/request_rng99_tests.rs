use super::*;
use effect_torch_graph::Device;
use effect_torch_runtime::StorageMetadata;

#[test]
#[ignore = "requires CUDA; request_rng99 shared family stream, isolation and cancellation"]
fn request_rng99_hardware_family_stream_isolation_and_cancellation() {
    assert!(RequestRng99::graphs_disabled());
    let node = |kind| Node::new(kind).unwrap();
    let input = node(NodeKind::Input {
        slot: 0,
        shape: vec![16],
        dtype: DType::F32,
        device: Device::Cuda(0),
        storage: StorageMetadata::dense(),
    });
    let noise = node(NodeKind::Uniform {
        lo: 0.,
        hi: 1.,
        shape: vec![16],
        dtype: DType::F32,
        device: Device::Cuda(0),
    });
    let root = node(NodeKind::Add { a: input, b: noise });
    let compile = |seed| {
        compile_with_options(
            vec![root.clone()],
            0,
            CompileOptions {
                random_seed: Some(seed),
                ..CompileOptions::default()
            },
        )
        .unwrap()
    };
    let initial = Arc::new(compile(0));
    let refinement = Arc::new(compile(0));
    let reference = compile(73);
    assert!(initial.request_rng99_admitted());
    assert!(!reference.request_rng99_admitted());
    let values = [CudaValue::from_host(
        CudaDevice::get(0).unwrap(),
        vec![16],
        DType::F32,
        &vec![1.; 16],
    )
    .unwrap()];
    let invoke = |program: &CudaExecutable,
                  rng: &RequestRng99,
                  bindings: &[CudaValue],
                  cancel: &CancellationFlag| {
        program.execute_inner(
            bindings,
            &[],
            None,
            cancel,
            Some(rng),
            None,
            None,
            None,
            None,
            None,
        )
    };
    let request = Arc::new(RequestRng99::new(73));
    let cancelled = CancellationFlag::new();
    cancelled.cancel();
    assert!(invoke(&initial, &request, &values, &cancelled).is_err());
    assert_eq!(request.runs.load(Ordering::Relaxed), 0);
    let mut retained = Vec::new();
    for program in [&initial, &refinement, &refinement, &initial] {
        let actual = invoke(program, &request, &values, &CancellationFlag::new()).unwrap();
        let expected = reference
            .execute(&values, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap();
        assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
        retained.push((actual, expected));
    }
    // Pre-dispatch template rejection must not consume a request ordinal.
    let fresh = RequestRng99::new(73);
    assert!(invoke(&reference, &fresh, &values, &CancellationFlag::new()).is_err());
    assert_eq!(fresh.runs.load(Ordering::Relaxed), 0);
    let first = invoke(&refinement, &fresh, &values, &CancellationFlag::new()).unwrap();
    assert_eq!(first[0].read_storage_bytes().unwrap(), retained[0].1);
    // Failure after admission consumes exactly one ordinal, as ordinary execution does.
    assert!(invoke(&initial, &request, &[], &CancellationFlag::new()).is_err());
    assert!(reference
        .execute(&[], &[], &CancellationFlag::new())
        .is_err());
    let recovered = invoke(&refinement, &request, &values, &CancellationFlag::new()).unwrap();
    let expected = reference
        .execute(&values, &[], &CancellationFlag::new())
        .unwrap();
    assert_eq!(
        recovered[0].read_storage_bytes().unwrap(),
        expected[0].read_storage_bytes().unwrap()
    );
    // Templates retain their independent original counters and seeds.
    let template_control = compile(0);
    assert_eq!(
        initial
            .execute(&values, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap(),
        template_control
            .execute(&values, &[], &CancellationFlag::new())
            .unwrap()[0]
            .read_storage_bytes()
            .unwrap()
    );
    for (actual, expected) in retained {
        assert_eq!(actual[0].read_storage_bytes().unwrap(), expected);
    }
    // Concurrent independent request families sharing the same immutable plan.
    let mut threads = Vec::new();
    for _ in 0..2 {
        let program = initial.clone();
        let bindings = values.to_vec();
        threads.push(std::thread::spawn(move || {
            program
                .execute_inner(
                    &bindings,
                    &[],
                    None,
                    &CancellationFlag::new(),
                    Some(&RequestRng99::new(73)),
                    None,
                    None,
                    None,
                    None,
                    None,
                )
                .unwrap()[0]
                .read_storage_bytes()
                .unwrap()
        }));
    }
    let expected = first[0].read_storage_bytes().unwrap();
    for thread in threads {
        assert_eq!(thread.join().unwrap(), expected);
    }
}
