from locust import HttpUser, task, between
import random

class PenguinUser(HttpUser):
    wait_time = between(1, 2)

    def random_penguin(self):
        return {
            "bill_length_mm": random.uniform(35, 50),
            "bill_depth_mm": random.uniform(13, 20),
            "flipper_length_mm": random.uniform(170, 230),
            "body_mass_g": random.uniform(2700, 6000),
            "island": random.choice(["Biscoe", "Dream", "Torgersen"]),
            "sex": random.choice(["male", "female"])
        }

    @task
    def predict(self):
        payload = self.random_penguin()
        self.client.post("/predict", json=payload)