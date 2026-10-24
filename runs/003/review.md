# review 003 (Critic)

Hypothesis (48x28 resolution raises recall without losing precision): NOT supported. REJECT: F1 0.7283 vs 0.7279 (+0.0004 < epsilon 0.002), result.json.
- Recall 0.744 -> 0.838 but precision 0.712 -> 0.644, flagged share 52.2% -> 65.1%, and the threshold moved from 0.63 to 0.51.
- Engineer reports OOF AUC 0.7977 vs 0.7983 and unchanged mean probabilities by class. So the ranking did not improve; the move along the PR curve came from the threshold choice, not better separation.
- Confident FN (p<0.3) went from 237 to 188, but p<0.1 went from 14 to 22. Confident FP went from 182 to 160. There is no clear gain on the dark/small-figure misses.
- Conclusion: spatial resolution of the grayscale median/MAD reference is not the bottleneck. The remaining confusion is likely about WHAT differs (normal object/character positions vary across normal frames vs anomalous content), which pixel z-scores at either resolution do not capture.
- Cost: extraction 450 s, run 137 s, and swap nearly full (~1.1-1.4 GB free). Any next idea must be lighter on memory.
- Status: 1 REJECT after a KEEP; best stays 002.
