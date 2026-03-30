# I Taught an AI to Stop Ignoring Shadows (Here's How)

In the last two months, I studied image augmentation methods in deep learning for computer vision. I was able to innovate and finish a project applying image augmentation to road feature detection from overhead imagery. This blog summarizes the project and what I achieved.

Honestly, I had no idea this would turn into such a cool problem. Let me walk you through it.

---

## 1. Wait, Why Is the AI Missing Stoplines?

So here's what I was trying to do. I was working with a deep learning model that looks at satellite photos and tries to find road markings — things like stoplines at intersections. Pretty straightforward, right?

Except the model kept messing up in one specific situation: **shadows**.

Whenever a building or a tree cast a shadow over a stopline, the AI just completely missed it. Like, it would do great on the sunny parts of the image, then completely blank out on anything in the shade. Look at Figure 1 — the green lines are what the model detected, and the red dots are the actual stoplines. In the shadowed area? Nothing.

| ![shadow issue 1](./materials/jpeg_images/shadow_issue_1.jpg) | ![shadow issue 2](./materials/jpeg_images/shadow_issue_2.jpg) |
| :----------------------------------------------------------- | :----------------------------------------------------------- |

**Figure 1.** Green lines = what the AI found. Red dots = where stoplines actually are. In the shadowed patches, the AI finds nothing.

### So... why is this happening?

When I dug into it, the answer was kind of obvious once I saw it. The AI learned from training data, and most of the training images were taken in bright, normal lighting. Shadowed stoplines? Barely in there. So the model just never really learned what they look like.

I counted it up in Table 1 and it was worse than I expected:

**Table 1. How Much of the Training Data is in Shadow?**

| Dataset    | Total Markings | Fully in Shadow | Partially in Shadow |
| ---------- | -------------: | --------------: | ------------------: |
| Stopline-1 | 6,615          | 375 (5.67%)     | 841 (12.71%)        |
| Stopline-2 | 1,851          | 88 (4.75%)      | 236 (12.75%)        |

Only about 17–19% of all road markings have any shadow on them at all. So the model basically never practiced on shadowed cases — no wonder it fails on them.

### Can't we just collect more data?

That was my first thought too. But even if we went out and grabbed more satellite photos, they'd still mostly be in normal lighting. That's just how aerial imagery works — shadows are always going to be the minority. You can't really fix it by collecting more of the same stuff.

My next thought was: okay, can I just use one of those copy-paste tricks? Like, crop a stopline from one image and paste it somewhere dark?

Here's the thing — road markings are really different from, say, a cat in a photo. You can recognize a cat even if you cut it out and drop it on a random background. But a stopline? It looks like a stopline *because* of where it is — at the end of a lane, on pavement, near an intersection. If you rip it out of context and paste it somewhere random, you're feeding the AI completely fake, misleading examples. The whole point of road markings is that their *surroundings* are part of how you identify them.

So that rules out a lot of the standard tricks. I needed something that could make the training images harder and more realistic — but without moving anything around or breaking the scene.

---

## 2. What Have Other People Tried?

Before I came up with my own approach, I looked at what researchers have done to fake shadows for AI training. Here's a quick rundown:

| Technique           | What It Does                                    | How Hard? |
| ------------------- | ----------------------------------------------- | --------- |
| Geometric Shading   | Just darkens one side of the image              | Easy      |
| Albumentations      | Popular library, has shadow settings built in   | Very Easy |
| OBA (satellite)     | Pastes cropped objects with fake shadows added  | Medium    |
| Gaussian Shadow     | Blob-shaped shadow (originally for ultrasound)  | Medium    |
| Pixel Height Maps   | Uses 3D geometry to compute shadow positions    | Hard      |
| Ray-Traced (COFFEE) | Full physics simulation in a game engine        | Very Hard |
| RSH                 | Random shadow and highlight patches             | Easy      |
| Action Recognition  | Big, heavy shadows to stress-test video AI      | Easy      |

The easy ones (like just darkening part of the image) look too fake for satellite photos — the AI would probably figure out they're not real. And the super realistic ones, like ray-tracing shadows in a simulator? Way overkill, and honestly way too complicated for what I needed.

Also, a lot of these methods assume you can just move objects around. For road markings, as I just explained, that doesn't work.

So I was looking for something in the middle: realistic enough to actually fool the AI, but not so complicated that it takes forever to build. And it had to work within the original scene — no moving things around.

---

## 3. My Solution: Just Fake the Shadows

My idea was pretty simple when I put it that way: *during training, add realistic fake shadows on top of the road images*. The model gets way more practice seeing shadowed markings, so it gets better at detecting them.

The whole pipeline has two parts. First, some one-time setup work before training. Then, the actual shadow generation that runs automatically on every training image.

- **One-time setup:** Study real shadows from satellite photos to figure out what they actually look like.
- **Every training image:** Generate a brand new fake shadow and paste it onto the image.

![Shadow augmentation pipeline](./materials/jpeg_images/shadow_aug_diagram.jpg)

**Figure 2.** The full pipeline. Left side is the one-time setup. Right side runs on every single training image.

### Step 1: First, Find the Real Shadows

To learn what shadows look like, I first needed to automatically find them in the training photos. I built a detector that checks the image in multiple color formats at the same time — kind of like how you might look at a problem from different angles to make sure you understand it. By combining those checks, it can reliably tell shadow pixels from normal ones.

![Shadow segmentation flowchart](./materials/jpeg_images/shadow_seg.jpg)

**Figure 3.** The shadow detection process — checking multiple color representations at once.

Here's what it produces. Black = shadow, everything else = normal lighting:

|           | Original Photo                               | Shadow Map                                              |
| --------- | -------------------------------------------- | ------------------------------------------------------- |
| Example 1 | ![](./materials/jpeg_images/54968_35510.jpg) | ![](./materials/jpeg_images/54968_35510_shadowmask.jpg) |
| Example 2 | ![](./materials/jpeg_images/55101_36113.jpg) | ![](./materials/jpeg_images/55101_36113_shadowmask.jpg) |

**Figure 4.** The black regions show what the detector identified as shadow. Looks pretty accurate!

### Step 2: Learn the Shadow "Color Recipe"

At first I thought I could just make pixels darker to fake a shadow. Turns out that looks really bad. Real shadows have a specific *color character* — they're not just dimmer, they have a certain hue and saturation too.

So I measured the color statistics (technically called HSV — hue, saturation, and brightness) of the real shadow pixels I found in Step 1. I calculated the average and how much it varies. Then when I generate a fake shadow, I randomly pick a color from within that measured range. That way every shadow looks a little different, but they all stay within the range of what real shadows actually look like.

![Shadow characterization flowchart](./materials/jpeg_images/shadow_properties.jpg)

**Figure 5.** How I measured the color properties of real shadows.

| Hue | Saturation | Brightness |
| :-: | :--------: | :--------: |
| ![](./materials/jpeg_images/cw_shadow_h.jpg) | ![](./materials/jpeg_images/cw_shadow_s.jpg) | ![](./materials/jpeg_images/cw_shadow_v.jpg) |

**Figure 6.** These are the color distributions I measured. When making a fake shadow, I randomly sample from these ranges.

### Step 3: Grab the Shadow Texture

Here's a detail I didn't think about at first: real shadows aren't perfectly smooth. There's slight grain and texture in them — from the pavement underneath, camera noise, atmospheric stuff. If my fake shadows are too smooth and clean, they'll look obviously fake.

To fix this, I extracted texture patterns from the real shadow regions. Basically I separated each patch into "smooth background" + "rough detail on top," then saved a bunch of those detail patches. When I make a fake shadow later, I grab one of these real texture patches and add it in. Makes a huge difference in how realistic it looks.

|                        Rough texture                         |                        Smooth texture                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src='./materials/jpeg_images/shadow_texture.jpg' width=400> | <img src='./materials/jpeg_images/shadow_texture_1.jpeg' width=400> |

**Figure 7.** Two texture patches I extracted from real shadow regions. These get reused when generating fake shadows.

### Step 4: Make a Random Shadow Shape

Real shadows have all kinds of weird irregular shapes depending on what's blocking the sun. I needed a way to generate random blob-like shapes that look natural.

I ended up using something called **Bézier curves**, which is actually the same math that graphic designers use to draw smooth curved shapes in tools like Illustrator. You place a bunch of random control points, and the curve smoothly connects through them. String a bunch of these curves together into a closed loop and you get a random irregular shape — kind of like a weird potato outline.



![](./used_image/equation_1.jpg)

(That formula looks scary but it's basically just: given some control points, draw a smooth curve through them.)

![Bézier curve example](./materials/jpeg_images/cub_bez_curve.jpg)

**Figure 8.** How a Bézier curve gets built from control points.

![Random shadow shapes](./materials/jpeg_images/random_shapes.jpg)

**Figure 9.** Some examples of randomly generated shadow outlines. Every single one is different.

### Step 5: Stick It All Together

Now I combine the three things — color recipe, texture patch, and random shape — into one fake shadow. Then I blend it onto the training image inside the shape boundary.

One nice thing: I don't have to change the labels at all. The labels record where the road markings are, and I'm not moving anything — just changing how the image looks. So the labels stay 100% valid.

|                         Shadow shape                         |                         Final result                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src='./materials/jpeg_images/emp_random_shape.jpg' width=400> | <img src='./materials/jpeg_images/emp_random_shadow.jpg' width=400> |

**Figure 10.** Left: the random shape mask. Right: the finished fake shadow composited onto the image.

| Shadow Generation | Shadow Application |
| :---------------: | :----------------: |
| ![](./materials/jpeg_images/shadow_gen.jpg) | ![](./materials/jpeg_images/shadow_apply.jpg) |

**Figure 11.** The full generation and application workflow.

Here's what it looks like on actual training images:

| Example 1 | Example 2 |
| :-------: | :-------: |
| ![](./materials/jpeg_images/aug_shadow.jpg) | ![](./materials/jpeg_images/aug_shadow_1.jpg) |

**Figure 12.** Fake shadows applied to stopline training images. Pretty convincing!

And it works on crosswalks too, without any changes to the method:

|                         Crosswalk 1                          |                         Crosswalk 2                          |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src='./materials/jpeg_images/cw_example.jpg' width=300> | <img src='./materials/jpeg_images/cw_example_2.jpg' width=600> |

**Figure 13.** Same method, applied to crosswalk images. No extra work needed.

### But Do the Fake Shadows Actually Fool Anyone?

I wanted to check if my synthetic shadows were realistic enough to actually matter. So I tried this: take a model that was trained with *no* shadow augmentation, and run it on images where I'd added fake shadows. If my shadows are good enough, that model should fail — the same way it fails on real shadows.

And it did! Look at Figure 14:

| Normal (no shadow) | After fake shadow — miss 1 | After fake shadow — miss 2 |
| :----------------: | :------------------------: | :------------------------: |
| ![](./materials/jpeg_images/54949_35511_baseline.jpg) | ![](./materials/jpeg_images/54949_35511_aug_shadow_degrade_1.jpg) | ![](./materials/jpeg_images/54949_35511_aug_shadow_degrade.jpg) |

**Figure 14.** The model detects fine on the original image (left), but misses detections after my fake shadows are added. That means the fakes are realistic enough to use as training data!

---

## 4. Okay, Did It Actually Work?

Short answer: yes, and more than I expected honestly.

I trained two versions of the detector — one normal, one with my shadow augmentation added — and tested both on the same set of held-out images neither had seen before.

**Table 2. Results: Normal Model vs. Shadow-Augmented Model**

| Model                 | Precision | Recall | F1    | False Alarms    | Missed Detections |
| --------------------- | --------: | -----: | ----: | --------------: | ----------------: |
| Normal (no aug)       | 0.952     | 0.918  | 0.935 | 61 / 1,268      | 104 / 1,266       |
| + Shadow Augmentation | **0.960** | **0.936** | **0.948** | 53 / 1,268 | 81 / 1,266 |

Here's what stands out to me:
- The augmented model **missed 22% fewer stoplines** (went from 104 missed down to 81)
- It had **13% fewer false alarms** too (61 down to 53)
- F1 score improved by 1.3 points

That false negative drop is the big one — that's exactly the problem I set out to fix. 22% fewer missed detections just from adding fake shadows during training. No changes to the model at all.

Here's the visual comparison:

|              Without shadow augmentation               |                With shadow augmentation                |
| :----------------------------------------------------: | :----------------------------------------------------: |
| ![](./materials/jpeg_images/trained_wo_shadow_agu.jpg) | ![](./materials/jpeg_images/trained_wz_shadow_agu.jpg) |

**Figure 15.** Left: normal model, misses detections in shadows. Right: augmented model, finds them.

I also ran this on a second stopline dataset and on crosswalk images. Across all those tests, the improvement was consistently somewhere between 2 and 5 percentage points in precision, recall, and F1. It wasn't a fluke — it kept working.

---

## 5. Why Does It Work? And Where Does It Break?

### Why it works (my theory)

I think there are three things going on:

1. **More reps on the hard stuff.** The model now sees shadowed markings all the time during training instead of almost never. It's like studying for the specific questions that are going to be on the test.

2. **Stopping the brightness fixation.** After seeing the same marking with and without shadow over and over, the model learns to focus on shape and location rather than just how bright something is. It stops being tricked by dark areas.

3. **Better calibration.** The model's internal confidence threshold for "is this a stopline?" gets tuned with more shadow examples. It's less likely to be uncertain and guess wrong.

The reason this actually transfers to real shadows (and not just fake ones) is that I measured the color stats from real satellite photos. The fake shadows look like real ones, so whatever robustness the model builds during training actually carries over.

### Where it breaks down

The method isn't perfect though, and I want to be honest about the failure cases:

- **Hard shadow edges.** My Bézier shapes make smooth, blobby outlines. But sometimes real shadows — like from the sharp corner of a building — have very hard, perfectly straight edges. There's a mismatch there that could confuse the model.

- **Shadow covering everything.** If a synthetic shadow is really dark and completely covers a road marking, not even a human could see it. Training on those examples is useless. I mostly avoid this by keeping the color sampling within a realistic range, but it can still happen.

- **Different cities or cameras.** The color statistics I measured are specific to the satellite camera and city I was working with. If someone used a different satellite or a city with very different pavement colors, they'd probably need to re-measure the shadow stats for their own data.

### How to tell if it's actually working

The easiest sanity check: run augmented images through a model trained *without* augmentation (like I showed in Figure 14). If that model doesn't struggle on the fake shadows, the shadows are too subtle to be useful.

Other things I'd check:
- Look at some of the shadow masks — are they actually finding the right regions, or are there big mistakes?
- Plot the brightness values of fake shadows vs. real ones to make sure they overlap
- Evaluate specifically on shadow regions to confirm the improvement is actually coming from shadow performance and not something else

---

## 6. What I Learned

This project was harder than I expected, but I learned a ton from it.

The biggest takeaway: **if your model hasn't seen enough examples of a hard case during training, it will fail on that case every time.** That sounds obvious, but the tricky part is that you can't always just collect more data to fix it. Sometimes the imbalance is built into how the world works — like how shadows are always going to be a minority in daytime aerial photos.

Shadow augmentation fixes this by generating realistic fake examples of exactly the hard case the model is struggling with. The numbers speak for themselves: 22% fewer missed detections, 13% fewer false alarms, and 2–5 percentage points of improvement across multiple datasets and road marking types. And all of this without changing a single thing about the model architecture.

What I'm most proud of is that I figured out *why* the standard tricks don't work here, and designed something that respects the actual structure of the problem. Road markings need context to be detectable. So the augmentation had to stay in-scene, and the shadows had to look real. Both of those constraints shaped every design decision.

I'll be honest — when I first started I wasn't sure this was going to work. Seeing it actually reduce missed detections by 22% was a really satisfying moment.

---

## References

**Geometric Region-Based Shading:**
Yadav, V. (2016). *An augmentation based deep neural network approach to learn human driving behavior.* Chatbots Life.

**Albumentations Library:**
Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., & Kalinin, A. A. (2020). Albumentations: Fast and flexible image augmentations. *Information, 11*(2).

**Object-Based Augmentation (OBA):**
Illarionova, S., Nesteruk, S., Shadrin, D., Ignatiev, V., Pukalchik, M., & Oseledets, I. (2021). Object-based augmentation for building semantic segmentation: Ventura and Santa Rosa case study. *CVF Open Access.*

**Gaussian Shadowing (Ultrasound):**
Tupper, A., & Gagné, C. (2024). Revisiting data augmentation for ultrasound images. *arXiv preprint.*

**Random Shadows and Highlights (RSH):**
Mazhar, O., & Kober, J. (2021). Random shadows and highlights: A new data augmentation method for extreme lighting conditions. *arXiv preprint arXiv:2101.05361.*

**Shadow Augmentation for Action Recognition:**
Ju, S., & Reibman, A. R. (2024). Shadow augmentation for handwashing action recognition: from synthetic to real datasets. *arXiv preprint arXiv:2410.03984.*

**Controllable Shadow Generation (Pixel Height Maps):**
Sheng, Y., Liu, Y., Zhang, J., Yin, W., Oztireli, A. C., Zhang, H., Lin, Z., Shechtman, E., & Benes, B. (2022). Controllable shadow generation using pixel height maps. *arXiv preprint arXiv:2207.05385.*

**Soft Shadow Network (SSG):**
Sheng, Y., Zhang, J., & Benes, B. (2021). SSN: Soft shadow network for image compositing. *CVPR.*

**COFFEE (Ray-Traced Synthetic Shadows):**
Zimmermann, A., Chung, S.-J., & Hadaegh, F. (2024). COFFEE: A shadow-resilient real-time pose estimator for unknown tumbling asteroids using sparse neural networks. *IAC-24-A3/IP/83.*



