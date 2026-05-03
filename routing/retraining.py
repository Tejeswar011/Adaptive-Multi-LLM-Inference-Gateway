from routing.train_router import train

RETRAIN_EVERY_N = 50


class RetrainingScheduler:

    def __init__(self, router, retrain_every=RETRAIN_EVERY_N):
        self.router = router
        self.retrain_every = retrain_every
        self.query_count = 0

    def on_query(self):
        self.query_count += 1

        if self.query_count % self.retrain_every == 0:
            print(f"Retraining router after {self.query_count} queries...")
            success = train()

            if success:
                self.router.reload_ml_model()
                print("Router model reloaded")
            else:
                print("Retraining skipped (not enough data or error)")
