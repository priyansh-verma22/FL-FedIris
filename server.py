# server.py -----------------------------------------------------
import os
import flwr as fl
from flwr.common import parameters_to_ndarrays          # ← convert helper
from flwr.server.strategy import FedAvg
from model import build_model                           # ← your model builder

# ---------- Custom strategy that writes weights into `model` ----------
class SaveModelStrategy(FedAvg):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model                              # keep a reference

    # override aggregate_fit so we can grab the final weights each round
    def aggregate_fit(self, rnd, results, failures):
        # aggregated_parameters is a *Parameters* object
        aggregated_parameters, metrics = super().aggregate_fit(
            rnd, results, failures
        )
        if aggregated_parameters is not None:
            # ✅ Convert Parameters → list[np.ndarray]
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            # ✅ Load into the Keras model
            self.model.set_weights(ndarrays)

        # Flower still needs us to return what FedAvg normally returns
        return aggregated_parameters, metrics

# ---------------------- main() ---------------------- #
def main():
    model = build_model(num_classes=74)                # build once

    strategy = SaveModelStrategy(
        model=model,
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=3,
        min_available_clients=3,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8081",
        config=fl.server.ServerConfig(num_rounds=20),
        strategy=strategy,
    )

    # after training, `model` now has the federated weights
    os.makedirs(r"C:\Users\priya\Desktop\Iris_Recognition\Gpt\results")
    model.save(r"C:\Users\priya\Desktop\Iris_Recognition\Gpt\results\global_model_final.h5")          # full model
    model.save_weights("results/global_model_final_weights.h5")  # weights only
    print("✅ Global model saved!")

if __name__ == "__main__":
    main()
