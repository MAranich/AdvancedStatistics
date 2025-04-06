
use AdvancedStatistics::{distribution_trait::Distribution, distributions::Normal::STD_NORMAL};

fn main() {

    let a: Vec<f64> = STD_NORMAL.quantile_multiple(&[0.1, 0.3, 0.6, 0.85]); 
    //let a: f64 = STD_NORMAL.quantile(0.55); 
    // println!("Final:       {:?}", a); 
    println!("{:?}", a); 
    
}