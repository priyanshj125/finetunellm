# Demo Video

## What to Record

The demo should show the complete fine-tuning pipeline running end to end in Google Colab.

## Recommended Structure (5 to 10 minutes)

### Part 1: Setup (1 minute)
Open the Colab notebook. Show the runtime type set to T4 GPU. Run Cell 1 (GPU check) and Cell 2 (dependency install). Confirm all libraries install without bitsandbytes errors.

### Part 2: Data Preparation (1 minute)
Run Cells 3 through 5. Show the dataset loading from HuggingFace and the curated Q&A samples. Show the final dataset size and train/test split.

### Part 3: Model and LoRA Setup (1 minute)
Run Cells 6 through 8. Show the model loading in fp16 and the LoRA adapter statistics. Highlight that only approximately 1% of parameters are trainable.

### Part 4: Training (2 to 3 minutes)
Run Cells 9 through 13. Show the training progress log with loss values decreasing. If training takes too long to show live, show a screenshot of completed training with the final loss value.

### Part 5: Evaluation (2 minutes)
Run Cell 15 to show the loss curve plot. Run Cells 17 through 20 to show the base vs fine-tuned comparison. Highlight at least one question where the fine-tuned model gives a noticeably better answer. Show the ROUGE and BLEU metric comparison chart.

### Part 6: Failure Cases (1 minute)
Run Cell 21 to show the failure case analysis. Demonstrate the model being asked for real-time data it cannot provide.

## Recording Tips

- Use Loom, OBS, or any screen recorder.
- Zoom in on code cells before running them so they are readable.
- Pause briefly on the final evaluation table to let viewers read the numbers.
- For the qualitative comparison, read one base answer and one fine-tuned answer aloud to highlight the improvement.

## Submission

Upload to YouTube (unlisted) or Google Drive and include the link in your submission form.
