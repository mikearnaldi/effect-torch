use super::tensor::Tensor;

fn pad1(x: &Tensor, padding: usize) -> Tensor {
    if padding == 0 {
        return x.clone();
    }
    let (n, c, l) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let zeros = Tensor::zeros(&[n, c, padding], x.dtype());
    Tensor::cat(&[&zeros, x, &zeros], 2)
}

fn pad2(x: &Tensor, padding: usize) -> Tensor {
    if padding == 0 {
        return x.clone();
    }
    let (n, c, h, w) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let zh = Tensor::zeros(&[n, c, padding, w], x.dtype());
    let x = Tensor::cat(&[&zh, x, &zh], 2);
    let zw = Tensor::zeros(&[n, c, h + 2 * padding, padding], x.dtype());
    Tensor::cat(&[&zw, &x, &zw], 3)
}

pub fn conv1d(x: &Tensor, w: &Tensor, stride: usize, padding: usize, dilation: usize, groups: usize) -> Tensor {
    let (n, c_in, l) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let (c_out, c_per, k) = (w.shape()[0], w.shape()[1], w.shape()[2]);
    assert_eq!(c_in, c_per * groups);
    let cout_per = c_out / groups;
    let x = pad1(x, padding);
    let l_out = (l + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
    let xc = x.contiguous();
    let wc = w.contiguous();
    let mut out = vec![0f64; n * c_out * l_out];
    let xf = xc.cast(crate::runtime::dtype::DType::F64);
    let wf = wc.cast(crate::runtime::dtype::DType::F64);
    let super::tensor::CpuBuffer::F64(xd) = &xf.buffer else { unreachable!() };
    let super::tensor::CpuBuffer::F64(wd) = &wf.buffer else { unreachable!() };
    let lp = l + 2 * padding;
    for b in 0..n {
        for g in 0..groups {
            for co in 0..cout_per {
                let oc = g * cout_per + co;
                for i in 0..l_out {
                    let mut acc = 0f64;
                    for ci in 0..c_per {
                        let ic = g * c_per + ci;
                        for kk in 0..k {
                            let pos = i * stride + kk * dilation;
                            if pos < lp {
                                acc += xd[(b * c_in + ic) * lp + pos] * wd[(oc * c_per + ci) * k + kk];
                            }
                        }
                    }
                    out[(b * c_out + oc) * l_out + i] = acc;
                }
            }
        }
    }
    Tensor::from_vec(out, vec![n, c_out, l_out]).cast(x.dtype())
}

pub fn conv2d(x: &Tensor, w: &Tensor, stride: usize, padding: usize, dilation: usize, groups: usize) -> Tensor {
    let (n, c_in, h, wd) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (c_out, c_per, kh, kw) = (w.shape()[0], w.shape()[1], w.shape()[2], w.shape()[3]);
    assert_eq!(c_in, c_per * groups);
    let cout_per = c_out / groups;
    let x = pad2(x, padding);
    let oh = (h + 2 * padding - dilation * (kh - 1) - 1) / stride + 1;
    let ow = (wd + 2 * padding - dilation * (kw - 1) - 1) / stride + 1;
    let xc = x.contiguous().cast(crate::runtime::dtype::DType::F64);
    let wc = w.contiguous().cast(crate::runtime::dtype::DType::F64);
    let super::tensor::CpuBuffer::F64(xd) = &xc.buffer else { unreachable!() };
    let super::tensor::CpuBuffer::F64(wgt) = &wc.buffer else { unreachable!() };
    let (hp, wp) = (h + 2 * padding, wd + 2 * padding);
    let mut out = vec![0f64; n * c_out * oh * ow];
    for b in 0..n {
        for g in 0..groups {
            for co in 0..cout_per {
                let oc = g * cout_per + co;
                for i in 0..oh {
                    for j in 0..ow {
                        let mut acc = 0f64;
                        for ci in 0..c_per {
                            let ic = g * c_per + ci;
                            for ky in 0..kh {
                                let py = i * stride + ky * dilation;
                                if py >= hp {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let px = j * stride + kx * dilation;
                                    if px >= wp {
                                        continue;
                                    }
                                    acc += xd[((b * c_in + ic) * hp + py) * wp + px]
                                        * wgt[((oc * c_per + ci) * kh + ky) * kw + kx];
                                }
                            }
                        }
                        out[((b * c_out + oc) * oh + i) * ow + j] = acc;
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, vec![n, c_out, oh, ow]).cast(x.dtype())
}

pub fn conv_transpose1d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let (n, c_in, l) = (x.shape()[0], x.shape()[1], x.shape()[2]);
    let (c_out_per_g, cin_per, k) = (w.shape()[1], w.shape()[0], w.shape()[2]);
    assert_eq!(c_in, cin_per * groups);
    let c_out = c_out_per_g * groups;
    let l_out = (l - 1) * stride - 2 * padding + dilation * (k - 1) + output_padding + 1;
    let xc = x.contiguous().cast(crate::runtime::dtype::DType::F64);
    let wc = w.contiguous().cast(crate::runtime::dtype::DType::F64);
    let super::tensor::CpuBuffer::F64(xd) = &xc.buffer else { unreachable!() };
    let super::tensor::CpuBuffer::F64(wgt) = &wc.buffer else { unreachable!() };
    let mut out = vec![0f64; n * c_out * l_out];
    for b in 0..n {
        for g in 0..groups {
            for ci in 0..cin_per {
                let ic = g * cin_per + ci;
                for i in 0..l {
                    let xv = xd[(b * c_in + ic) * l + i];
                    if xv == 0.0 {
                        continue;
                    }
                    for co in 0..c_out_per_g {
                        let oc = g * c_out_per_g + co;
                        for kk in 0..k {
                            let pos = i * stride + kk * dilation;
                            if pos >= padding && pos - padding < l_out {
                                out[(b * c_out + oc) * l_out + pos - padding] +=
                                    xv * wgt[((ic * c_out_per_g + co) * k) + kk];
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, vec![n, c_out, l_out]).cast(x.dtype())
}

pub fn conv_transpose2d(
    x: &Tensor,
    w: &Tensor,
    stride: usize,
    padding: usize,
    output_padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let (n, c_in, h, wd) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (c_out_per_g, cin_per, kh, kw) = (w.shape()[1], w.shape()[0], w.shape()[2], w.shape()[3]);
    assert_eq!(c_in, cin_per * groups);
    let c_out = c_out_per_g * groups;
    let oh = (h - 1) * stride - 2 * padding + dilation * (kh - 1) + output_padding + 1;
    let ow = (wd - 1) * stride - 2 * padding + dilation * (kw - 1) + output_padding + 1;
    let xc = x.contiguous().cast(crate::runtime::dtype::DType::F64);
    let wc = w.contiguous().cast(crate::runtime::dtype::DType::F64);
    let super::tensor::CpuBuffer::F64(xd) = &xc.buffer else { unreachable!() };
    let super::tensor::CpuBuffer::F64(wgt) = &wc.buffer else { unreachable!() };
    let mut out = vec![0f64; n * c_out * oh * ow];
    for b in 0..n {
        for g in 0..groups {
            for ci in 0..cin_per {
                let ic = g * cin_per + ci;
                for i in 0..h {
                    for j in 0..wd {
                        let xv = xd[((b * c_in + ic) * h + i) * wd + j];
                        if xv == 0.0 {
                            continue;
                        }
                        for co in 0..c_out_per_g {
                            let oc = g * c_out_per_g + co;
                            for ky in 0..kh {
                                let py = i * stride + ky * dilation;
                                if py < padding || py - padding >= oh {
                                    continue;
                                }
                                for kx in 0..kw {
                                    let px = j * stride + kx * dilation;
                                    if px < padding || px - padding >= ow {
                                        continue;
                                    }
                                    out[((b * c_out + oc) * oh + (py - padding)) * ow + (px - padding)] +=
                                        xv * wgt[((ic * c_out_per_g + co) * kh + ky) * kw + kx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, vec![n, c_out, oh, ow]).cast(x.dtype())
}

pub fn conv2d_backward_w(
    x: &Tensor,
    g: &Tensor,
    kernel: [usize; 2],
    out_channels: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
) -> Tensor {
    let (n, c_in, _, _) = (x.shape()[0], x.shape()[1], x.shape()[2], x.shape()[3]);
    let (_, _, oh, ow) = (g.shape()[0], g.shape()[1], g.shape()[2], g.shape()[3]);
    let (kh, kw) = (kernel[0], kernel[1]);
    let c_per = c_in / groups;
    let cout_per = out_channels / groups;
    let x = pad2(x, padding);
    let (hp, wp) = (x.shape()[2], x.shape()[3]);
    let xc = x.contiguous().cast(crate::runtime::dtype::DType::F64);
    let gc = g.contiguous().cast(crate::runtime::dtype::DType::F64);
    let super::tensor::CpuBuffer::F64(xd) = &xc.buffer else { unreachable!() };
    let super::tensor::CpuBuffer::F64(gd) = &gc.buffer else { unreachable!() };
    let mut out = vec![0f64; out_channels * c_per * kh * kw];
    for gidx in 0..groups {
        for co in 0..cout_per {
            let oc = gidx * cout_per + co;
            for ci in 0..c_per {
                let ic = gidx * c_per + ci;
                for ky in 0..kh {
                    for kx in 0..kw {
                        let mut acc = 0f64;
                        for b in 0..n {
                            for i in 0..oh {
                                let py = i * stride + ky * dilation;
                                if py >= hp {
                                    continue;
                                }
                                for j in 0..ow {
                                    let px = j * stride + kx * dilation;
                                    if px >= wp {
                                        continue;
                                    }
                                    acc += gd[((b * out_channels + oc) * oh + i) * ow + j]
                                        * xd[((b * c_in + ic) * hp + py) * wp + px];
                                }
                            }
                        }
                        out[((oc * c_per + ci) * kh + ky) * kw + kx] = acc;
                    }
                }
            }
        }
    }
    Tensor::from_vec(out, vec![out_channels, c_per, kh, kw]).cast(x.dtype())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::cpu::CpuBuffer;

    fn f32_data(t: &Tensor) -> Vec<f32> {
        let CpuBuffer::F32(v) = &t.buffer else { panic!() };
        v.as_slice().to_vec()
    }

    #[test]
    fn conv1d_basic() {
        let x = Tensor::from_vec(vec![1f32, 2., 3., 4.], vec![1, 1, 4]);
        let w = Tensor::from_vec(vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv1d(&x, &w, 1, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 3]);
        assert_eq!(f32_data(&y), vec![3., 5., 7.]);
    }

    #[test]
    fn conv2d_basic() {
        let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let w = Tensor::from_vec(vec![1f32, 0., 0., 1.], vec![1, 1, 2, 2]);
        let y = conv2d(&x, &w, 1, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 2, 2]);
        assert_eq!(f32_data(&y), vec![6., 8., 12., 14.]);
    }

    #[test]
    fn conv_transpose1d_basic() {
        let x = Tensor::from_vec(vec![1f32, 2.], vec![1, 1, 2]);
        let w = Tensor::from_vec(vec![1f32, 1.], vec![1, 1, 2]);
        let y = conv_transpose1d(&x, &w, 2, 0, 0, 1, 1);
        assert_eq!(y.shape(), &[1, 1, 4]);
        assert_eq!(f32_data(&y), vec![1., 1., 2., 2.]);
    }

    #[test]
    fn conv2d_backward_w_basic() {
        let x = Tensor::from_vec((1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3]);
        let g = Tensor::from_vec(vec![1f32, 1., 1., 1.], vec![1, 1, 2, 2]);
        let dw = conv2d_backward_w(&x, &g, [2, 2], 1, 1, 0, 1, 1);
        assert_eq!(f32_data(&dw), vec![12., 16., 24., 28.]);
    }
}
