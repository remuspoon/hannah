# V3 Improvements

- **decouple canonify prompt from sentence extraction**
    - We do both with one step right now, maybe it will do better if we separate those processes.
- **backstory consistency**
    - the autobiography still has inconsistencies. Upgrade to thinking model? And if we give more sentences maybe it'll have more sentences to composite a background from?
- **protective scenes**
    - we assume user behaves. We did not add any protective scenes (asking user what is python, asking user to solve a math problem)
- **add *thinking* utterances for counselor scenes**
    - Thinking utterances only exists on memory scenes. Could add thinking to counselor scenes to improve outputs.
- **Scale autobiograph so hannah utterances are all real written utterances**
    - We had 1391 sentences in this example, using 274 posts. Assuming min 15 hannah utterances, max 30 hannah utterances, at 650 conversations that will be 9750 - 19500 sentences that I need to extract. At around 5 sentences per post, that's 1950 - 3900 posts.
- **add crisis**
    - Currently Hannah does not exhibit active suicidal ideation / Crisis. Not sure how that will work...
    - Supplementary to crisis: Add tool that hannah can call to "end the call" (i.e. kill herself)
- **normalise json structures**
    - self explanatory. The downsides of claude code :(