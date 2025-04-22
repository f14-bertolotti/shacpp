val directions = setOf("north", "south", "east", "west", "north_east", "north_west", "south_east", "south_west")
val objects = setOf("tree", "rock", "home")

mas {
    beliefs {
        for (d in directions) {
            belief { "direction"($d) }.meaning {
                "${arg[0]} is a $functor"
            }
        }
        for (o in objects) {
            belief { "object"($o) }.meaning {
                "${arg[0]} is an object"
            }
        }
        belief { "direction"("here") }.meaning {
            "${arg[0]} denotes the null $functor w.r.t. the agent's current location"
        }
        admissible {
            "obstacle"("Direction") {
                "there is an $functor to the ${arg[0]}"
            }
        }
        admissible {
            "there_is"("Object", "Direction") {
                "there is a ${arg[0]} to the ${arg[1]}"
            }
        }
    }
    actions {
        action("move", 1) {
            // yakta code here
        }.meaning {
            "move in the given direction: ${arg[0]}"
        }
    }
    goals {
        achieve { "reach"("Object") }.meaning {
            "reach a situation where ${arg[0]} is in the position of the agent (i.e. there_is(${arg[0]}, here))"
        }
        // if there were further non-initial goals which are admissible:
        // admissible { ... }
    }
}