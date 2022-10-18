const SEED: u64 = 1491742;
const INIT_RANGE: std::ops::Range<FloatType> = -0.01..0.01;
const PARAMS_FILE: &str = "target/params";

const LR: FloatType = 0.03;
const WEIGHT_DECAY: FloatType = 0.0;
const HIDDEN_SIZE: usize = 100;
const INPUT_LEN: usize = 25;
const BETA_1: FloatType = 0.9;
const BETA_2: FloatType = 0.999;

const LOG_INTERVAL: u64 = 10;

const GEN_START: &[u8; 5] = b"hello";
const GEN_LEN: usize = 800;

#[cfg(test)]
const EPSILON: FloatType = 1e-2;
#[cfg(test)]
const THRESHOLD: FloatType = 1e-4;

use std::{
    collections::HashMap,
    env,
    f32::consts::E,
    fs::{self, File},
    io::Read,
    mem::{self, size_of},
    time::Instant,
};

use libc::{c_float, c_int};
use rand::{prelude::SmallRng, Rng, SeedableRng};

type FloatType = f32;

#[repr(C)]
#[allow(dead_code)]
enum CblasLayout {
    CblasRowMajor = 101,
    CblasColMajor = 102,
}

#[repr(C)]
#[allow(dead_code)]
enum CblasTranspose {
    CblasNoTrans = 111,
    CblasTrans = 112,
    CblasConjTrans = 113,
}

#[allow(dead_code)]
extern "C" {
    fn cblas_sgemm(
        layout: CblasLayout,
        trans_a: CblasTranspose,
        trans_b: CblasTranspose,
        m: c_int,
        n: c_int,
        k: c_int,
        alpha: c_float,
        a: *const c_float,
        lda: c_int,
        b: *const c_float,
        ldb: c_int,
        beta: c_float,
        c: *mut c_float,
        ldc: c_int,
    );

    fn cblas_saxpy(
        n: c_int,
        alpha: c_float,
        x: *const c_float,
        inc_x: c_int,
        y: *mut c_float,
        inc_y: c_int,
    );

    fn cblas_sasum(n: c_int, x: *const c_float, inc_x: c_int) -> c_float;

    fn vsMul(n: c_int, a: *const c_float, b: *const c_float, r: *mut c_float);

    fn vsTanh(n: c_int, a: *const c_float, r: *mut c_float);
}

fn dot(a: &[FloatType], b: &[FloatType], output: &mut [FloatType], a_width_b_height: usize) {
    #[cfg(not(feature = "no_blas"))]
    {
        assert!(a.len() % a_width_b_height == 0);
        assert!(b.len() % a_width_b_height == 0);
        assert!(output.len() == a.len() / a_width_b_height * (b.len() / a_width_b_height));

        unsafe {
            cblas_sgemm(
                CblasLayout::CblasRowMajor,
                CblasTranspose::CblasNoTrans,
                CblasTranspose::CblasNoTrans,
                (a.len() / a_width_b_height) as c_int,
                (b.len() / a_width_b_height) as c_int,
                a_width_b_height as c_int,
                1.0,
                a.as_ptr(),
                a_width_b_height as c_int,
                b.as_ptr(),
                (b.len() / a_width_b_height) as c_int,
                1.0,
                output.as_mut_ptr(),
                (b.len() / a_width_b_height) as c_int,
            );
        }
    }

    #[cfg(feature = "no_blas")]
    {
        let a_height = a.len() / a_width_b_height;
        let b_width = b.len() / a_width_b_height;

        for y in 0..a_height {
            for x in 0..b_width {
                output[y * b_width + x] += a[y * a_width_b_height..(y + 1) * a_width_b_height]
                    .iter()
                    .zip(b[x..].iter().step_by(b_width))
                    .map(|(a, b)| a * b)
                    .sum::<FloatType>();
            }
        }
    }
}

fn vec_mul(a: &[FloatType], b: &[FloatType], output: &mut [FloatType]) {
    #[cfg(not(feature = "no_blas"))]
    {
        assert!(a.len() == b.len());
        assert!(a.len() == output.len());

        unsafe {
            vsMul(
                a.len() as c_int,
                a.as_ptr(),
                b.as_ptr(),
                output.as_mut_ptr(),
            );
        }
    }

    #[cfg(feature = "no_blas")]
    {
        output
            .iter_mut()
            .zip(a)
            .zip(b)
            .for_each(|((o, a), b)| *o = a * b);
    }
}

fn vec_add(a: &[FloatType], output: &mut [FloatType]) {
    #[cfg(not(feature = "no_blas"))]
    {
        assert!(a.len() == output.len());

        unsafe {
            cblas_saxpy(a.len() as c_int, 1.0, a.as_ptr(), 1, output.as_mut_ptr(), 1);
        }
    }

    #[cfg(feature = "no_blas")]
    {
        output.iter_mut().zip(a).for_each(|(o, a)| *o += a);
    }
}

fn vec_tanh(input: &[FloatType], output: &mut [FloatType]) {
    #[cfg(not(feature = "no_blas"))]
    {
        assert!(input.len() == output.len());

        unsafe {
            vsTanh(input.len() as c_int, input.as_ptr(), output.as_mut_ptr());
        }
    }

    #[cfg(feature = "no_blas")]
    {
        output
            .iter_mut()
            .zip(input)
            .for_each(|(o, i)| *o = i.tanh());
    }
}

fn abs_sum(x: &[FloatType]) -> FloatType {
    #[cfg(not(feature = "no_blas"))]
    {
        unsafe { cblas_sasum(x.len() as c_int, x.as_ptr(), 1) }
    }

    #[cfg(feature = "no_blas")]
    {
        x.iter().map(|v| v.abs()).sum()
    }
}

fn sigmoid(x: FloatType) -> FloatType {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(arr: &mut [FloatType]) {
    let max = arr.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    arr.iter_mut().for_each(|v| *v = (*v - max).exp());
    let output_sum = abs_sum(arr);
    arr.iter_mut().for_each(|v| *v /= output_sum);
}

fn forward(
    num_input_nodes: usize,
    num_hidden_nodes: usize,
    (
        update_input_weight,
        update_state_weight,
        update_bias,
        reset_input_weight,
        reset_state_weight,
        reset_bias,
        activ_input_weight,
        activ_state_weight,
        activ_bias,
        output_bias,
        output_weight,
    ): (
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
        &[FloatType],
    ),
    input_at_step: &[FloatType],
    state_at_step: &[FloatType],
    update_sum: &mut [FloatType],
    update: &mut [FloatType],
    reset_sum: &mut [FloatType],
    reset: &mut [FloatType],
    reset_x_state: &mut [FloatType],
    activ_sum: &mut [FloatType],
    activ: &mut [FloatType],
    hidden: &mut [FloatType],
    output: &mut [FloatType],
) {
    update_sum.copy_from_slice(update_bias);
    dot(
        &input_at_step,
        update_input_weight,
        update_sum,
        num_input_nodes,
    );
    dot(
        state_at_step,
        update_state_weight,
        update_sum,
        num_hidden_nodes,
    );
    update
        .iter_mut()
        .zip(update_sum)
        .for_each(|(a, b)| *a = sigmoid(*b));

    reset_sum.copy_from_slice(reset_bias);
    dot(
        &input_at_step,
        reset_input_weight,
        reset_sum,
        num_input_nodes,
    );
    dot(
        state_at_step,
        reset_state_weight,
        reset_sum,
        num_hidden_nodes,
    );
    reset
        .iter_mut()
        .zip(reset_sum)
        .for_each(|(a, b)| *a = sigmoid(*b));

    vec_mul(&reset, state_at_step, reset_x_state);

    activ_sum.copy_from_slice(activ_bias);
    dot(
        &input_at_step,
        activ_input_weight,
        activ_sum,
        num_input_nodes,
    );
    dot(
        &reset_x_state,
        activ_state_weight,
        activ_sum,
        num_hidden_nodes,
    );
    vec_tanh(&activ_sum, activ);

    vec_mul(&update, &activ, hidden);
    hidden
        .iter_mut()
        .zip(update)
        .zip(state_at_step)
        .for_each(|((h, u), s)| *h += (1.0 - *u) * s);

    output.copy_from_slice(output_bias);

    dot(hidden, output_weight, output, num_hidden_nodes);
    softmax(output);
}

fn split_params(
    num_input_nodes: usize,
    num_hidden_nodes: usize,
    num_output_nodes: usize,
    params: &[FloatType],
) -> (
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
    &[FloatType],
) {
    let (a, next) = params.split_at(num_input_nodes * num_hidden_nodes);
    let (b, next) = next.split_at(num_hidden_nodes * num_hidden_nodes);
    let (c, next) = next.split_at(num_hidden_nodes);
    let (d, next) = next.split_at(num_input_nodes * num_hidden_nodes);
    let (e, next) = next.split_at(num_hidden_nodes * num_hidden_nodes);
    let (f, next) = next.split_at(num_hidden_nodes);
    let (g, next) = next.split_at(num_input_nodes * num_hidden_nodes);
    let (h, next) = next.split_at(num_hidden_nodes * num_hidden_nodes);
    let (i, next) = next.split_at(num_hidden_nodes);
    let (j, next) = next.split_at(num_output_nodes);
    let (k, _) = next.split_at(num_hidden_nodes * num_output_nodes);

    (a, b, c, d, e, f, g, h, i, j, k)
}

fn network(
    num_steps: usize,
    num_input_nodes: usize,
    num_hidden_nodes: usize,
    num_output_nodes: usize,
    input: &[usize],
    output: &mut [FloatType],
    params: &[FloatType],
    correct: Option<&[usize]>,
    grad: Option<&mut [FloatType]>,
    state: &mut [FloatType],
) -> FloatType {
    let (
        update_input_weight,
        update_state_weight,
        update_bias,
        reset_input_weight,
        reset_state_weight,
        reset_bias,
        activ_input_weight,
        activ_state_weight,
        activ_bias,
        output_bias,
        output_weight,
    ) = split_params(num_input_nodes, num_hidden_nodes, num_output_nodes, params);

    let mut loss = 0.0;
    let mut input_at_step = vec![0.0; num_input_nodes];
    let mut hidden = vec![0.0; num_steps * num_hidden_nodes];

    let mut update_sum = vec![0.0; num_hidden_nodes];
    let mut update = vec![0.0; num_steps * num_hidden_nodes];
    let mut reset_sum = vec![0.0; num_hidden_nodes];
    let mut reset = vec![0.0; num_steps * num_hidden_nodes];
    let mut reset_x_state = vec![0.0; num_steps * num_hidden_nodes];
    let mut activ_sum = vec![0.0; num_hidden_nodes];
    let mut activ = vec![0.0; num_steps * num_hidden_nodes];

    for step in 0..num_steps {
        let output = &mut output[step * num_output_nodes..(step + 1) * num_output_nodes];
        let update = &mut update[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
        let reset = &mut reset[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
        let reset_x_state =
            &mut reset_x_state[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
        let activ = &mut activ[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
        input_at_step.fill(0.0);
        input_at_step[input[step]] = 1.0;

        let state_at_step: &[FloatType];
        let hidden_at_step;
        if step == 0 {
            state_at_step = state;
            hidden_at_step = &mut hidden[..num_hidden_nodes];
        } else {
            let split = hidden.split_at_mut(step * num_hidden_nodes);
            state_at_step = &split.0[(step - 1) * num_hidden_nodes..];
            hidden_at_step = &mut split.1[..num_hidden_nodes];
        }

        forward(
            num_input_nodes,
            num_hidden_nodes,
            (
                &update_input_weight,
                &update_state_weight,
                &update_bias,
                &reset_input_weight,
                &reset_state_weight,
                &reset_bias,
                &activ_input_weight,
                &activ_state_weight,
                &activ_bias,
                &output_bias,
                &output_weight,
            ),
            &input_at_step,
            &state_at_step,
            &mut update_sum,
            update,
            &mut reset_sum,
            reset,
            reset_x_state,
            &mut activ_sum,
            activ,
            hidden_at_step,
            output,
        );

        if let Some(correct) = correct {
            loss -= output[correct[step]].log(E);
        }
    }

    if let (Some(correct), Some(grad)) = (correct, grad) {
        grad.fill(0.0);
        let (grad_update_input_weight, next) =
            grad.split_at_mut(num_input_nodes * num_hidden_nodes);
        let (grad_update_state_weight, next) =
            next.split_at_mut(num_hidden_nodes * num_hidden_nodes);
        let (grad_update_bias, next) = next.split_at_mut(num_hidden_nodes);
        let (grad_reset_input_weight, next) = next.split_at_mut(num_input_nodes * num_hidden_nodes);
        let (grad_reset_state_weight, next) =
            next.split_at_mut(num_hidden_nodes * num_hidden_nodes);
        let (grad_reset_bias, next) = next.split_at_mut(num_hidden_nodes);
        let (grad_activ_input_weight, next) = next.split_at_mut(num_input_nodes * num_hidden_nodes);
        let (grad_activ_state_weight, next) =
            next.split_at_mut(num_hidden_nodes * num_hidden_nodes);
        let (grad_activ_bias, next) = next.split_at_mut(num_hidden_nodes);
        let (grad_output_bias, next) = next.split_at_mut(num_output_nodes);
        let (grad_output_weight, _) = next.split_at_mut(num_hidden_nodes * num_output_nodes);

        let mut grad_softmax_input = vec![0.0; num_output_nodes];
        let mut grad_hidden = vec![0.0; num_hidden_nodes];
        let mut grad_activ_sum = vec![0.0; num_hidden_nodes];
        let mut grad_reset_x_state = vec![0.0; num_hidden_nodes];
        let mut grad_update_sum = vec![0.0; num_hidden_nodes];
        let mut grad_reset_sum = vec![0.0; num_hidden_nodes];
        let mut grad_state = vec![0.0; num_hidden_nodes];

        for step in (0..num_steps).rev() {
            let output = &output[step * num_output_nodes..(step + 1) * num_output_nodes];
            let update = &update[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
            let reset = &reset[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
            let reset_x_state =
                &reset_x_state[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
            let activ = &activ[step * num_hidden_nodes..(step + 1) * num_hidden_nodes];
            input_at_step.fill(0.0);
            input_at_step[input[step]] = 1.0;

            let state_at_step: &[FloatType];
            let hidden_at_step;
            if step == 0 {
                state_at_step = state;
                hidden_at_step = &mut hidden[..num_hidden_nodes];
            } else {
                let split = hidden.split_at_mut(step * num_hidden_nodes);
                state_at_step = &split.0[(step - 1) * num_hidden_nodes..];
                hidden_at_step = &mut split.1[..num_hidden_nodes];
            }

            grad_softmax_input.copy_from_slice(output);
            grad_softmax_input[correct[step]] -= 1.0;

            grad_hidden.copy_from_slice(&grad_state);
            dot(
                output_weight,
                &grad_softmax_input,
                &mut grad_hidden,
                num_output_nodes,
            );

            grad_activ_sum
                .iter_mut()
                .zip(&grad_hidden)
                .zip(update)
                .zip(activ)
                .for_each(|(((a, b), c), d)| *a = b * c * (1.0 - d * d));

            grad_reset_x_state.fill(0.0);
            dot(
                activ_state_weight,
                &grad_activ_sum,
                &mut grad_reset_x_state,
                num_hidden_nodes,
            );

            grad_update_sum
                .iter_mut()
                .zip(&grad_hidden)
                .zip(state_at_step)
                .zip(update)
                .for_each(|(((a, b), c), d)| *a = b * -1.0 * c * (1.0 - d) * d);
            grad_update_sum
                .iter_mut()
                .zip(&grad_hidden)
                .zip(activ)
                .zip(update)
                .for_each(|(((a, b), c), d)| *a += b * c * (1.0 - d) * d);

            grad_reset_sum
                .iter_mut()
                .zip(&grad_reset_x_state)
                .zip(state_at_step)
                .zip(reset)
                .for_each(|(((a, b), c), d)| *a = b * c * (1.0 - d) * d);

            vec_mul(&reset, &grad_reset_x_state, &mut grad_state);
            grad_state
                .iter_mut()
                .zip(update)
                .zip(&grad_hidden)
                .for_each(|((a, b), c)| *a += (1.0 - b) * c);
            dot(
                &update_state_weight,
                &grad_update_sum,
                &mut grad_state,
                num_hidden_nodes,
            );
            dot(
                &reset_state_weight,
                &grad_reset_sum,
                &mut grad_state,
                num_hidden_nodes,
            );

            dot(
                &input_at_step,
                &grad_update_sum,
                grad_update_input_weight,
                1,
            );

            dot(state_at_step, &grad_update_sum, grad_update_state_weight, 1);

            vec_add(&grad_update_sum, grad_update_bias);

            dot(&input_at_step, &grad_reset_sum, grad_reset_input_weight, 1);

            dot(state_at_step, &grad_reset_sum, grad_reset_state_weight, 1);

            vec_add(&grad_reset_sum, grad_reset_bias);

            dot(&input_at_step, &grad_activ_sum, grad_activ_input_weight, 1);

            dot(&reset_x_state, &grad_activ_sum, grad_activ_state_weight, 1);

            vec_add(&grad_activ_sum, grad_activ_bias);

            vec_add(&grad_softmax_input, grad_output_bias);

            dot(hidden_at_step, &grad_softmax_input, grad_output_weight, 1);
        }
    }

    state.copy_from_slice(&hidden[(num_steps - 1) * num_hidden_nodes..]);

    return loss;
}

fn num_params(num_input_nodes: usize, num_hidden_nodes: usize, num_output_nodes: usize) -> usize {
    (num_input_nodes * num_hidden_nodes + num_hidden_nodes * num_hidden_nodes + num_hidden_nodes)
        * 3
        + num_hidden_nodes * num_output_nodes
        + num_output_nodes
}

#[allow(dead_code)]
fn sgd(params: &mut [FloatType], grad: &mut [FloatType]) {
    assert!(params.len() == grad.len());
    unsafe {
        if WEIGHT_DECAY != 0.0 {
            cblas_saxpy(
                params.len() as c_int,
                WEIGHT_DECAY,
                params.as_ptr(),
                1,
                grad.as_mut_ptr(),
                1,
            );
        }
        cblas_saxpy(
            params.len() as c_int,
            -LR,
            grad.as_ptr(),
            1,
            params.as_mut_ptr(),
            1,
        );
    }
}

#[allow(dead_code)]
fn adam(
    params: &mut [FloatType],
    grad: &[FloatType],
    m: &mut [FloatType],
    v: &mut [FloatType],
    epoch: FloatType,
) {
    params
        .iter_mut()
        .zip(grad)
        .zip(m)
        .zip(v)
        .for_each(|(((p, &g), m), v)| {
            *m = BETA_1 * *m + (1.0 - BETA_1) * g;
            *v = BETA_2 * *v + (1.0 - BETA_2) * g * g;
            let mhat = *m / (1.0 - BETA_1.powf(epoch + 1.0));
            let vhat = *v / (1.0 - BETA_2.powf(epoch + 1.0));
            *p -= LR * mhat / (vhat.sqrt() + f32::EPSILON);
        });
}

fn random_slice(x: &mut [FloatType], rng: &mut SmallRng) {
    for v in x {
        *v = rng.gen_range(INIT_RANGE);
    }
}

fn read_text() -> Vec<u8> {
    let mut file = File::open("my_text_file").unwrap();
    let mut contents = Vec::new();
    file.read_to_end(&mut contents).unwrap();

    contents
}

fn gen_char_map(input: &[u8]) -> (HashMap<u8, usize>, HashMap<usize, u8>) {
    let mut char_to_num = HashMap::new();
    let mut num_to_char = HashMap::new();
    let mut num_chars = 0;
    for v in input {
        if !char_to_num.contains_key(v) {
            char_to_num.insert(*v, num_chars);
            num_to_char.insert(num_chars, *v);
            num_chars += 1;
        }
    }

    (char_to_num, num_to_char)
}

fn map_text(text: &[u8], char_to_num: &HashMap<u8, usize>) -> Vec<usize> {
    let mut result = vec![0; text.len()];
    result
        .iter_mut()
        .zip(text)
        .for_each(|(r, i)| *r = *char_to_num.get(i).unwrap());
    result
}

fn choose_in_dist(input: &[FloatType], num: FloatType) -> usize {
    let mut sum = 0.0;
    for (i, v) in input.iter().enumerate() {
        sum += v;
        if num <= sum {
            return i;
        }
    }
    0
}

fn gen_text(
    params: &[FloatType],
    char_to_num: &HashMap<u8, usize>,
    num_to_char: &HashMap<usize, u8>,
    gen_start: &[u8],
    num_chars: usize,
    gen_len: usize,
    rng: &mut SmallRng,
) -> Vec<u8> {
    let mut text = gen_start.to_vec();
    let mut input = vec![0.0; num_chars];
    let mut prev_state = vec![0.0; HIDDEN_SIZE];
    let mut next_state = vec![0.0; HIDDEN_SIZE];
    let mut output = vec![0.0; num_chars];

    let mut prev_input_idx = 0;
    let mut processed_start_chars = 0;

    while text.len() < gen_len {
        let input_idx;
        if processed_start_chars < gen_start.len() {
            input_idx = *char_to_num.get(&text[processed_start_chars]).unwrap();
            processed_start_chars += 1;
        } else {
            input_idx = *char_to_num.get(text.last().unwrap()).unwrap();
        }
        input[prev_input_idx] = 0.0;
        input[input_idx] = 1.0;
        prev_input_idx = input_idx;

        forward(
            num_chars,
            HIDDEN_SIZE,
            split_params(num_chars, HIDDEN_SIZE, num_chars, params),
            &input,
            &prev_state,
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut vec![0.0; HIDDEN_SIZE],
            &mut next_state,
            &mut output,
        );
        mem::swap(&mut prev_state, &mut next_state);

        if processed_start_chars >= gen_start.len() {
            let predicted_idx = choose_in_dist(&output, rng.gen_range(0.0..1.0));
            text.push(*num_to_char.get(&predicted_idx).unwrap());
        }
    }

    text
}

#[test]
fn test() {
    let num_steps = 4;
    let num_input_nodes = 12;
    let num_hidden_nodes = 13;
    let num_output_nodes = 14;
    let mut rng = SmallRng::seed_from_u64(SEED);

    let num_params = num_params(num_input_nodes, num_hidden_nodes, num_output_nodes);
    let mut params = vec![0.0; num_params];
    random_slice(&mut params, &mut rng);
    let mut input = vec![0; num_steps];
    for v in &mut input {
        *v = rng.gen_range(0..num_input_nodes);
    }
    let mut correct = vec![0; num_steps];
    for v in &mut correct {
        *v = rng.gen_range(0..num_output_nodes);
    }
    let mut output = vec![0.0; num_steps * num_output_nodes];
    let state = vec![1.0; num_hidden_nodes];
    let mut grad = vec![0.0; num_params];

    network(
        num_steps,
        num_input_nodes,
        num_hidden_nodes,
        num_output_nodes,
        &input,
        &mut output,
        &params,
        Some(&correct),
        Some(&mut grad),
        &mut state.clone(),
    );

    let mut count = 0;
    for i in 0..params.len() {
        params[i] += EPSILON;
        let up = network(
            num_steps,
            num_input_nodes,
            num_hidden_nodes,
            num_output_nodes,
            &input,
            &mut output,
            &params,
            Some(&correct),
            None,
            &mut state.clone(),
        );
        params[i] -= EPSILON;

        params[i] -= EPSILON;
        let down = network(
            num_steps,
            num_input_nodes,
            num_hidden_nodes,
            num_output_nodes,
            &input,
            &mut output,
            &params,
            Some(&correct),
            None,
            &mut state.clone(),
        );
        params[i] += EPSILON;

        let actual = (up - down) / (2.0 * EPSILON);
        let diff = (actual - grad[i]).abs();
        if diff >= THRESHOLD {
            println!(
                "{} {} {} {} (actual, predicted, diff, i)",
                actual, grad[i], diff, i
            );
            count += 1;
        }
    }

    assert!(count == 0, "{} errors", count);
}

fn get_params(rng: &mut SmallRng, num_params: usize) -> Vec<FloatType> {
    let mut result = vec![0.0; num_params];
    let bytes = fs::read(PARAMS_FILE);

    match bytes {
        Ok(b) => {
            for (i, v) in result.iter_mut().enumerate() {
                let num_bytes = &b[i * size_of::<FloatType>()..(i + 1) * size_of::<FloatType>()];
                *v = FloatType::from_le_bytes(num_bytes.try_into().unwrap());
            }
        }
        Err(_) => {
            random_slice(&mut result, rng);
        }
    };

    result
}

fn write_params(params: &[FloatType]) {
    let size = size_of::<FloatType>();
    let mut bytes = vec![0; params.len() * size];
    for i in 0..params.len() {
        bytes[i * size..(i + 1) * size].copy_from_slice(&params[i].to_le_bytes());
    }

    fs::write(PARAMS_FILE, &bytes).unwrap();
}

fn main() {
    let mut rng = SmallRng::seed_from_u64(SEED);

    let text = read_text();
    let (char_to_num, num_to_char) = gen_char_map(&text);
    let num_chars = char_to_num.keys().len();
    let mapped_text = map_text(&text, &char_to_num);

    let num_params = num_params(num_chars, HIDDEN_SIZE, num_chars);
    let mut params = get_params(&mut rng, num_params);

    let args: Vec<String> = env::args().collect();
    if args.len() > 3 && args[1] == "gen" {
        let result = gen_text(
            &params,
            &char_to_num,
            &num_to_char,
            args[3].as_bytes(),
            num_chars,
            args[2].parse().unwrap(),
            &mut rng,
        );
        let result = String::from_utf8(result).unwrap();
        println!("{}", result);
        return;
    }

    let mut output = vec![0.0; INPUT_LEN * num_chars];
    let mut state = vec![0.0; HIDDEN_SIZE];
    let mut grad = vec![0.0; num_params];
    let mut loss = 0.0;

    let mut step = 0;
    let mut input_ptr = 0;
    let mut log_time = Instant::now();
    let mut log_steps = 0;

    loop {
        let input = &mapped_text[input_ptr..input_ptr + INPUT_LEN];
        let correct = &mapped_text[input_ptr + 1..input_ptr + 1 + INPUT_LEN];

        loss += network(
            INPUT_LEN,
            num_chars,
            HIDDEN_SIZE,
            num_chars,
            input,
            &mut output,
            &params,
            Some(correct),
            Some(&mut grad),
            &mut state,
        );
        assert!(loss.is_normal());

        sgd(&mut params, &mut grad);

        step += 1;
        log_steps += 1;
        input_ptr += INPUT_LEN;
        if input_ptr + 1 + INPUT_LEN > mapped_text.len() {
            input_ptr = 0;
            state.fill(0.0);
        }

        if log_time.elapsed().as_secs() > LOG_INTERVAL {
            write_params(&params);

            println!("====={} at {}=====", loss / log_steps as FloatType, step);

            let gen_bytes = gen_text(
                &params,
                &char_to_num,
                &num_to_char,
                GEN_START,
                num_chars,
                GEN_LEN,
                &mut rng,
            );
            println!("{}", String::from_utf8_lossy(&gen_bytes));

            loss = 0.0;
            log_time = Instant::now();
            log_steps = 0;
        }
    }
}
