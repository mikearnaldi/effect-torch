//! Joined ordinary expert branches. Grouped cuBLAS never enters this capture.
use super::{Bf16GemmPlan, CudaDevice, CUBLAS_WORKSPACE_BYTES};
use crate::cublas::expert_graphs::Key;
use cudarc::driver::CudaEvent;

#[cfg(test)]
thread_local! {
    static FAIL_CAPTURE_AFTER_FIRST_SUBMISSION: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

impl CudaDevice {
    pub(super) fn submit_ordinary_branch(
        &self,
        partitions: &[Vec<(Bf16GemmPlan, u64, u64, u64)>],
        worker_start: usize,
        workspace: u64,
        ready: &CudaEvent,
    ) -> Result<Vec<CudaEvent>, String> {
        let workers = &self.expert_cublas[worker_start..worker_start + partitions.len()];
        let origin = workers[0].stream();
        // Every graph launch needs only its origin's readiness dependency.
        // Eager admissions fork all children from that same origin below.
        origin.wait(ready).map_err(|e| e.to_string())?;
        self.expert_branch_graphs
            .lock()
            .map_err(|_| "ordinary branch graph cache lock poisoned")?
            .execute_with_events(
                origin,
                Key::partitioned(partitions, worker_start, workspace),
                |_capturing, events| {
                    let mut forked = 0;
                    let mut result = Ok(());
                    {
                        events.push(origin.record_event(None).map_err(|e| e.to_string())?);
                        for worker in &workers[1..] {
                            match worker.stream().wait(&events[0]) {
                                Ok(()) => forked += 1,
                                Err(error) => {
                                    result = Err(error.to_string());
                                    break;
                                }
                            }
                        }
                    }
                    if result.is_ok() {
                        for (index, (worker, groups)) in workers.iter().zip(partitions).enumerate()
                        {
                            // SAFETY: invocation leases and the outer submission fence
                            // cover every input/output/workspace until origin completion.
                            result = unsafe {
                                worker.gemm_bf16_sequence_eager(
                                    groups,
                                    workspace
                                        + ((index + worker_start) * CUBLAS_WORKSPACE_BYTES) as u64,
                                )
                            };
                            #[cfg(test)]
                            if _capturing
                                && index == 0
                                && FAIL_CAPTURE_AFTER_FIRST_SUBMISSION
                                    .with(|flag| flag.replace(false))
                            {
                                result = Err("injected ordinary branch capture failure".into());
                            }
                            if result.is_err() {
                                break;
                            }
                        }
                    }
                    // Join every successfully forked stream, including on failure.
                    // The cache guard then ends capture before the outer submission
                    // fence synchronizes workers. Retain events through that abort.
                    for worker in &workers[1..1 + forked] {
                        match worker.stream().record_event(None) {
                            Ok(event) => {
                                let joined = origin.wait(&event).map_err(|e| e.to_string());
                                events.push(event);
                                if result.is_ok() {
                                    result = joined;
                                }
                            }
                            Err(error) => {
                                if result.is_ok() {
                                    result = Err(error.to_string());
                                }
                            }
                        }
                    }
                    result
                },
            )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::CudaValue;
    use cudarc::driver::DevicePtr;
    use effect_torch_runtime::DType;

    #[test]
    #[ignore = "requires CUDA, EXPERT_BRANCH_GRAPHS=1 and GROUPED_EXACT=1"]
    fn joined_ordinary_branches_replay_and_abort_with_concurrent_grouped_work() {
        assert!(std::env::var("EFFECT_TORCH_CUDA_EXPERT_BRANCH_GRAPHS").is_ok_and(|v| v == "1"));
        assert!(std::env::var("EFFECT_TORCH_CUDA_GROUPED_EXACT").is_ok_and(|v| v == "1"));
        let device = CudaDevice::get(0).unwrap();
        let (n, k, rows) = (1408, 2816, 33);
        let data = |count: usize, seed: usize| {
            (0..count)
                .map(|i: usize| (((i * 1664525 + seed) >> 8) % 255usize) as f64 / 128.0 - 1.0)
                .collect::<Vec<_>>()
        };
        let weight =
            CudaValue::from_host(device.clone(), vec![n, k], DType::BF16, &data(n * k, 313))
                .unwrap();
        let inputs = (0..7)
            .map(|i| {
                CudaValue::from_host(
                    device.clone(),
                    vec![rows, k],
                    DType::BF16,
                    &data(rows * k, i * 997),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let outputs = || {
            (0..7)
                .map(|_| {
                    CudaValue::from_host(
                        device.clone(),
                        vec![rows, n],
                        DType::BF16,
                        &vec![0.; rows * n],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>()
        };
        let expected = outputs();
        let mut actual = outputs();
        let workspace_bytes = super::super::EXPERT_BLAS_STREAMS * CUBLAS_WORKSPACE_BYTES
            + super::super::EXPERT_GROUPED_POINTER_BYTES;
        let workspace = unsafe { device.stream.alloc::<u8>(workspace_bytes) }.unwrap();
        let (address, guard) = workspace.device_ptr(&device.stream);
        let mut retained = None;
        for iteration in 0..10 {
            if iteration == 4 {
                retained = Some((
                    actual
                        .iter()
                        .map(|v| v.readback().unwrap())
                        .collect::<Vec<_>>(),
                    actual,
                ));
                actual = outputs();
            }
            for (index, input) in inputs.iter().enumerate() {
                let bytes = crate::value::dense_bytes_from_host(
                    &data(rows * k, iteration * 91991 + index * 997),
                    DType::BF16,
                );
                let mut buffer = input.buffer.as_ref().clone();
                device.stream.memcpy_htod(&bytes, &mut buffer).unwrap();
            }
            let groups = [2, 3, 4, 5, 17, 33, 1]
                .into_iter()
                .enumerate()
                .map(|(index, m)| {
                    let plan = Bf16GemmPlan {
                        m,
                        n,
                        k,
                        batch: 1,
                        stride_x: m * k,
                        stride_weight: n * k,
                        stride_out: m * n,
                    };
                    unsafe {
                        device
                            .cublas
                            .gemm_bf16(
                                plan,
                                true,
                                inputs[index].storage_address(),
                                weight.storage_address(),
                                expected[index].storage_address(),
                                false,
                                address,
                            )
                            .unwrap();
                    }
                    (
                        plan,
                        inputs[index].storage_address(),
                        weight.storage_address(),
                        actual[index].storage_address(),
                    )
                })
                .collect::<Vec<_>>();
            if iteration == 5 {
                FAIL_CAPTURE_AFTER_FIRST_SUBMISSION.with(|flag| flag.set(true));
            }
            let result =
                unsafe { device.grouped_gemm_bf16(&groups, address, true, workspace_bytes) };
            if iteration == 5 {
                assert_eq!(
                    result.unwrap_err(),
                    "injected ordinary branch capture failure"
                );
                continue;
            }
            let _completed = result.unwrap();
            for (a, e) in actual.iter().zip(&expected) {
                assert_eq!(
                    a.readback().unwrap(),
                    e.readback().unwrap(),
                    "iteration {iteration}"
                );
            }
        }
        let (snapshots, retained) = retained.unwrap();
        for (value, snapshot) in retained.iter().zip(snapshots) {
            assert_eq!(value.readback().unwrap(), snapshot);
        }
        device.stream.synchronize().unwrap();
        let cache = device.expert_branch_graphs.lock().unwrap();
        assert_eq!(cache.captures, 2);
        assert!(cache.hits >= 4);
        drop(guard);
    }
}
