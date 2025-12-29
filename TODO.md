# Stuff TODO

- [x] Finished task
- [ ] Unfinished task

- [x] Use proper training phases
- [x] Group train logs by experiment
- [ ] Add final loss and MAC to filename after finishing training
- [ ] Separate finished runs and unfinished runs in the log directory
- [ ] Fix args, their loading, printing
- [ ] Do back to back encodeing + decoding and measure actual bpd
- [ ] For that: We need to have pixel by pixel model decoding
- [x] Need to save model parameters
- [x] Need to quantize and save the latents
- [ ] Enable full arithmetic coding in all cases when training is finished
- [ ] Quad tree like splitting of trained image encoder + small finetune
- [x] Add torch.compile to speed up training
- [ ] Fix torch.compile issues with broken graphs

## Plan of action for today:

- [x] Move presents to config folder
- [x] Move args to config folder
- [ ] Add encode/decode validation after training is finished
- [ ] Refactor the lossless encode file to be cleaner -> should have these parts:
  - [ ] imports
  - [ ] initialization (args, model, image)
- if time allows:
  - [ ] make models modular to allow easy printing of layers
  - [ ] add torch compile to speed up training
- [ ] add warmups
- [ ] fix image encoder manager logs
- [ ] fix test function output logs
- [ ] move computation of probs to separate module inside coolchic 
