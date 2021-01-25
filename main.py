from obstacle_avoidance import main_obs as obs_main
import foraging.main_fora as for_main
from prey_predator import main_pred as pred_main

assignment = "prey_predator"
data_package = "data20.pkl"
labels_package = "labels20.pkl"
epochs = 10
batches = 16
mlp_hidden_nodes = 100
validation = 0.15
testing = 0.15
learning_rate = 0.01

if __name__ == "__main__":
    if assignment == "obstacle_avoidance":
        obs_main.main()

    elif assignment == "foraging":
        for_main.main(data_package=data_package,
                      labels_package=labels_package,
                      epochs=epochs,
                      batches=batches,
                      mlp_hidden_nodes=mlp_hidden_nodes,
                      validation=validation,
                      testing=testing,
                      learning_rate=learning_rate)

    elif assignment == "prey_predator":
        pred_main.main()