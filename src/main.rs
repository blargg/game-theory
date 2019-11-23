use ndarray::*;

fn main() {
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
    let game = Game::new(pd_grid).unwrap();
    for player in game.players() {
        let actions: Vec<_> = game.strategies(player).collect();
        println!("player {}, has actions {:?}", player, actions);
    }

    let tell_dominates = game.strictly_dominates(0, 0, 1);
    println!("p1 tell dominates: {}", tell_dominates);
}

type Player = usize;
type Strategy = usize;

#[derive(Debug)]
struct Game {
    grid: ArrayD<f32>,
}


impl Game {
    fn new(payoff_grid: ArrayD<f32>) -> Option<Game> {
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

    fn players(&self) -> impl Iterator<Item=Player> {
        0..(self.grid.shape().len() - 1)
    }

    fn strategies(&self, player: Player) -> impl Iterator<Item=Strategy> {
        0..(self.grid.shape()[player])
    }

    // Determines if s1 strictly dominates s2 for the given player
    // This means that for all other players choices, s1 gives a better payoff than s2
    fn strictly_dominates(&self, player: Player, s1: Strategy, s2: Strategy) -> bool {
        let payoff_axis = Axis(self.grid.shape().len() - 1);
        let player_payoffs = self
            .grid
            .index_axis(payoff_axis, player);
        let p1 = player_payoffs
            .index_axis(Axis(player), s1);
        println!("p1: {:?}", p1);
        let p2 = player_payoffs
            .index_axis(Axis(player), s2);
        println!("p2: {:?}", p2);
        let mut comp = ArrayD::from_elem(p1.shape(), false);
        Zip::from(&mut comp)
            .and(p1)
            .and(p2)
            .apply(|c, v1, v2| {
                *c = v1 > v2;
            });
        println!("comparison: {:?}", comp);
        comp.iter().all(|i| *i)
    }
}
