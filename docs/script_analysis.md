Here's a more natural and student-like version of your explanation:

---

## **Overview**  
We compared two scripts, **`model_eval.ipynb`** and **`run_models.ipynb`**, both used in the PIXIU framework for evaluating financial models. The old script was mostly about setup, while the new one is actually responsible for running the models and evaluating their performance.  

## **Key Differences**  
The biggest difference between the two scripts is their purpose. **`model_eval.ipynb`** was mainly used to install dependencies, set up the environment, and make sure everything was configured correctly before running any models. It was more like a prep tool than something that did actual work. **`run_models.ipynb`**, on the other hand, is where the actual benchmarking happens. It runs the models and tasks, processes results, and handles evaluations.  

Another big change is the **execution time**. The new script runs multiple evalautions with large batch sizes, making it much slower. The old script didn’t have this problem since it wasn’t running anything heavy—it just got everything ready. This means the new script is more functional, but also way more time-consuming, so we might need to look into performance improvements down the line.  
