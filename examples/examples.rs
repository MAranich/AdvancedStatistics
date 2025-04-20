use AdvancedStatistics::{distribution_trait::Distribution, distributions::StudentT::StudentT};

fn main() {
    let dof: f64 = 30.0;
    let t_distr: StudentT = StudentT::new(dof).unwrap();

    println!("Degrees of freedom: {}", dof);
    //let points: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
    let points: [f64; 11] = [
        0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99, 0.995, 0.975, 0.999, 0.9995,
    ];
    let q_points: Vec<f64> = t_distr.quantile_multiple(&points);
    for (i, p) in points.iter().enumerate() {
        println!("{}: \t{}", p, q_points[i]);
    }
}
