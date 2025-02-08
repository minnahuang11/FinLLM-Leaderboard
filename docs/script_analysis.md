## Overview
Comparing the old script (`model_eval.ipynb`) and the new one (`run_models.ipynb`). Both are part of the PIXIU framework for financial model evaluation, but they serve different roles. The old script was mainly used for setup, while the new script is focused on actually running and evaluating models.

## Key Differences
The biggest difference between the two is their purpose. The old script (`model_eval.ipynb`) was mostly about getting everything ready—it installed dependencies, set up the environment, and made sure everything was configured correctly before running any models. On the other hand, the new script (`run_models.ipynb`) actually runs the evaluations, making it the more active script. It includes specific commands to execute models and process results, while the old script was more like a setup guide that didn’t directly handle any tasks.

Another major difference is how long they take to run. The new script is noticeably slower because it runs multiple evaluations, each involving large batch sizes. This can make execution drag on, which is something to keep in mind when working with it. The old script didn’t have this issue since it wasn’t actually running any models—it just made sure everything was ready to go. So while both scripts are necessary, the old script was more about preparation, and the new script is where the real work happens.

Overall, the switch from `model_eval.ipynb` to `run_models.ipynb` represents a shift from setup to execution. The new script is more functional but also more time-consuming, so performance optimizations might be needed in the future.

