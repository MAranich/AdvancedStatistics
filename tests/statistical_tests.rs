use AdvancedStatistics::hypothesis::*;
use AdvancedStatistics::samples::*;
use assert_approx_eq::assert_approx_eq;

#[test]
fn t_test_test_1() {
    struct Round {
        data: Vec<f64>,
        null_mean: f64,
        hyp: Hypothesis,
        p_val: f64,
    }

    let rounds = [
        Round {
            data: vec![
                1.692, -0.070, 0.446, 2.292, -0.532, -0.186, -0.806, 2.888, 0.143, 0.820, 0.000,
                1.025, 0.298, 3.535, 0.032, 1.476, 1.209, 1.029, -0.745, 1.369, -0.141, -0.413,
                0.722, 0.510, -0.951, -0.164, 1.599, 1.163, 0.940, 0.649,
            ],
            null_mean: 0.0,
            hyp: Hypothesis::TwoTailed,
            p_val: 0.002039,
        },
        Round {
            data: vec![
                1.81267509,
                1.47946564,
                1.77264520,
                -0.44931665,
                -1.93698732,
                -0.50434139,
                -1.24863565,
                -1.28081648,
                1.96360507,
                -0.03323779,
                -1.40229678,
                -2.08196497,
            ],
            null_mean: 0.0,
            hyp: Hypothesis::TwoTailed,
            p_val: 0.7262,
        },
        Round {
            data: vec![
                -0.6489526, 3.9924671, -6.5579178, 4.7170022, 6.1731676, -1.3148294, -4.4146563,
                4.1561860, 1.1071276, -2.9148294, -1.7266905, 4.1359327, 3.0533137, 1.2340260,
                1.9694276, 3.6605833,
            ],
            null_mean: 0.0,
            hyp: Hypothesis::RightTail,
            p_val: 0.1358,
        },
        Round {
            data: vec![
                1.9804378,
                -2.5613539,
                2.6765804,
                -2.8449638,
                -2.7801385,
                -7.6233881,
                -0.3844971,
                5.2449107,
                -12.7083661,
                -2.4007930,
                4.1365391,
                1.3760967,
                -1.6537029,
                -4.1980103,
                -5.4968773,
                3.3820374,
                -1.2464542,
                1.1191590,
                -6.1347894,
                -5.3228133,
                -4.3453640,
                -0.6708169,
                -6.7356384,
                -2.5304418,
                -2.7141527,
                -1.4215870,
                0.4287888,
                5.8056787,
                8.2846899,
                5.4608863,
                -5.3197362,
                7.4535238,
                -1.0317579,
                -5.8861046,
                -10.4957019,
                3.8086446,
                -2.4017578,
                7.1700340,
                -9.3510711,
                7.0992387,
                6.8746684,
                -0.5925622,
                -0.3781905,
                -0.2620539,
                -4.8902803,
            ],
            null_mean: 0.0,
            hyp: Hypothesis::RightTail,
            p_val: 0.8898,
        },
    ];

    for round in rounds {
        let mut sampl: Samples = unsafe { Samples::new_move_uncheched(round.data) };
        let result: TestResult = t_test()
            .data(&mut sampl)
            .hypothesys(round.hyp)
            .null_mean(round.null_mean)
            .call()
            .expect("The inputs should be valid. ");

        assert_approx_eq!(round.p_val, result.p(), 0.0001_f64);
    }
}
