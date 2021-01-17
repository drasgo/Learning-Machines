import robobo
import obstacle_avoidance.main as obs_main
import foraging.main as for_main

assignment = "obstacle_avoidance"

if __name__ == "__main__":
    if assignment == "obstacle_avoidance":
        obs_main.main()
    elif assignment == "foraging":
        for_main.main()
    elif assignment == "prey_predator":
        pass