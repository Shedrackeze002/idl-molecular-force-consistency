# Late learning-rate outcome notes

Saved-only descriptive analysis of the verified once-only experiment. Each arm resumed the identical update-6,000 model and Adam state for 4,000 additional updates; only the prescribed learning-rate schedule changed. One seed and fitted examples only.

| Schedule | Fitting energy MSE remaining | Fitting force MSE remaining | Fitted-anchor energy remaining | Fitted-anchor force remaining | Joint gate |
|---|---:|---:|---:|---:|---|
| constant | 17.7114% | 32.4699% | 7.1096% | 30.6088% | False |
| late_decay | 19.1199% | 34.4558% | 7.3632% | 32.6289% | False |

Both schedules failed the unchanged requirement that energy AND force MSE fall below 10% of their original update-0 baselines on both fitting and fitted-anchor supports. Fitted-anchor energy passes individually; fitted-anchor force and both fitting metrics fail.

At update 10,000 the reduced-rate arm has 7.95% higher fitting energy MSE and 6.12% higher fitting force MSE than the constant-rate arm. This particular schedule did not improve the prescribed endpoint within the fixed budget. The result does not reject all possible learning-rate schedules.

The reduced-rate arm has a smaller range of sampled energy errors. These ranges include learning trend and sparse observations; they do not quantify continuous smoothness or rule out spikes between scores. Full observed local maxima and corresponding force errors are saved in the JSON.

The last 500 updates still reduce fitting force MSE by 2.04% (constant) and 1.68% (late decay). Small changes, especially with smaller update sizes, do not prove convergence.

At the endpoint, the squared global mean energy residual accounts for about 0.042% of fitting energy MSE in the constant arm and 0.000029% in the reduced-rate arm. A global offset therefore does not explain most endpoint energy error; group-specific offsets and earlier spike mechanisms remain unresolved. No recentering was applied to the gates.

No held-out evaluation, GPU training, main consistency intervention or shape-recovery comparison occurred. Full pilot remains pending. Suggested next experiment: a separately reviewed controlled representation-capacity check, followed by GPU parity before a broader pilot; no new experiment is authorized by this note.

## Transient spike and recovery

The constant-rate arm reached fitting energy MSE 1.4918430381672636 eV² at update 8,001, or 39.76470431227181 times the original update-0 energy MSE. At 7,900 the fraction was 0.1922721030, at 8,000 it was 24.6933246510, and at 8,100 it recovered to 0.1923267405. Force MSE at 8,001 rose to 0.7409307860 of the original baseline. All eight fitting groups show high energy MSE at this saved observation. The constant schedule did not change at 8,000/8,001. Its verified 8,001 update oracle passed against the accumulated full gradient and independent Adam calculation, with maximum absolute difference 1.4210854715202004e-14; this supports correct execution of this step, not an explanation of the spike.

The reduced-rate arm's observed energy fraction spans 0.1911994186–0.2037762112 across all 43 saved scores, with no comparable sampled spike. Scores are sparse, except the prescribed boundary check; no claim is made about unsampled updates or general stability. Endpoint error and transient behavior must both be reported, because the constant arm still ends with lower energy/force errors. Preserved linear and logarithmic figures show the full peak without hiding the lower-error trajectory.
