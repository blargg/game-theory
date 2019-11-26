use ndarray::*;

pub type Player = usize;
pub type Strategy = usize;

#[derive(Debug)]
pub struct Game {
    grid: ArrayD<f32>,
}


impl Game {
    pub fn new(payoff_grid: ArrayD<f32>) -> Option<Game> {
        let shape = payoff_grid.shape();
        let l = shape.len() - 1;
        if shape[l] == l {
            Some(
                Game {
                    grid: payoff_grid,
                }
            )
        } else {
            None
        }
    }

    pub fn players(&self) -> impl Iterator<Item=Player> {
        0..(self.grid.shape().len() - 1)
    }

    pub fn strategies(&self, player: Player) -> impl Iterator<Item=Strategy> {
        0..(self.grid.shape()[player])
    }

    /// Determines if s1 strictly dominates s2 for the given player
    /// This means that for all other players choices, s1 gives a better payoff than s2
    pub fn strictly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        self.compare_strats(|v1, v2| v1 > v2, player, s1, s2)
    }

    pub fn weakly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        self.compare_strats(|v1, v2| v1 >= v2, player, s1, s2)
    }


    fn compare_strats<F: Fn(f32, f32) -> bool>(&self, comparison: F, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let payoff_axis = Axis(self.grid.shape().len() - 1);
        let player_payoffs = self
            .grid
            .index_axis(payoff_axis, player);
        let p1 = player_payoffs
            .index_axis(Axis(player), s1);
        let p2 = player_payoffs
            .index_axis(Axis(player), s2);
        let mut comp = ArrayD::from_elem(p1.shape(), false);
        Zip::from(&mut comp)
            .and(p1)
            .and(p2)
            .apply(|c, v1, v2| {
                *c = comparison(*v1, *v2);
            });
        comp.iter().all(|i| *i)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn prisoners_dilemma() -> Game {
        let pd_grid = ArrayD::from_shape_fn(vec![2,2,2], |dim| {
            10.0 - match (dim[0], dim[1], dim[2]) {
                (0, 0, _) => 5.0,
                (1, 0, 0) => 7.0,
                (1, 0, 1) => 0.0,
                (0, 1, 0) => 0.0,
                (0, 1, 1) => 7.0,
                (1, 1, _) => 2.0,
                _ => 100.0,
            }
        });

        Game::new(pd_grid).unwrap()
    }

    #[test]
    fn test_strictly_dominates() {
        let pd = prisoners_dilemma();
        assert!(pd.strictly_dominates(0,0,1), "Action 0 (telling) should strictly dominate action 1");
        assert!(!pd.strictly_dominates(0,1,0), "Action 1 (keeping quiet) should not strictly dominate action 0");
    }
}
