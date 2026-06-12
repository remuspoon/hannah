# V3 Improvements

- **increase data size from 22 examples.**
    - Currently we only select posts with 6/6 regex tags. Could try lowering filter to 5/6 to increase amount of posts.
- **decouple canonify prompt from sentence extraction**
    - We do both with one step right now, maybe it will do better if we separate those processes.
- **backstory consistency**
    - the autobiography still has inconsistencies. Upgrade to thinking model? And if we give more sentences maybe it'll have more sentences to composite a background from?