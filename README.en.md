[简体中文](README.md) | **English**

# AI Avatar Live Assistant (ai-live-local)

The goal is a Windows desktop app that runs entirely on the local machine: it takes a streamer's own video and voice, produces a digital host that answers viewer comments in real time, and feeds the picture into Douyin Live Companion (抖音直播伴侣).

> **Current conclusion (2026-11-25): neither of the two lip-sync methods tested is feasible under this project's target conditions.** There is no usable version and no installer. The tests and the reasons are described below.

## Target conditions

A lip-sync method is feasible only if it meets all four:

| Condition | Requirement |
|---|---|
| Hardware | One RTX 3060 12GB, with the language model, speech synthesis and lip sync all running locally |
| Input | The streamer's own video (portrait, half or full body), not a single photo |
| Real time | 25 fps without stalls, and the avatar starts speaking within a few seconds of a comment |
| Picture quality | No visible blur around the mouth; viewers must not spot the synthesis at a glance |

## Test setup

- Test machine: RTX 3080 Ti 12GB, Xeon Gold 6248 (6 cores), 32 GB RAM, Windows Server 2022. **This is not the target machine.**
- An RTX 3060 has roughly half the compute of a 3080 Ti. Every "scaled to 3060" figure below uses that ratio; there are no measurements on a real 3060.
- Material: a 1080x1920 video of a person talking (12.5 seconds, face about 460x560 pixels) and Chinese speech in a cloned voice.

## Method 1: MuseTalk 1.5

**How it works**: only the mouth and chin are regenerated and pasted back into the original frame; the rest of the picture is the original pixels. The model outputs a 256x256 face.

This is the approach the project has used so far, and the real-time pipeline (`core/live.py`) runs end to end: comments arrive, the language model streams a reply, cloned speech is synthesized in chunks, lip sync is generated incrementally per audio chunk, and a clock-driven player outputs 25 fps.

**Measurements** (language model, speech synthesis and lip sync running together; 5 comments; 4 runs):

| Metric | Result |
|---|---|
| Lip-sync speed (GPU part, running alone) | about 23 ms per frame, about 43 fps |
| Scaled to 3060 | about 21 fps, below 25 fps |
| Average response latency (comment to first spoken frame) | 1.03 to 1.13 s |
| Stalls while speaking | in 3 of 4 runs, 0.20 to 0.36 s during the first reply |
| Peak GPU memory (all three models) | about 9.5 GB |

**What was done to sharpen the mouth**, all in `engines/lipsync.py` and `engines/enhance.py`:

1. Blending narrowed from the whole lower face to the mouth and chin.
2. A small super-resolution network (Real-ESRGAN compact) on the mouth, about 3 ms per frame.
3. Skin texture from the original video (pores, stubble) added back onto the generated area.
4. Sharpening of the lip area only (texture cannot be added there, so it previously had no enhancement at all).
5. The super-resolved result is resized once to the face box, removing an intermediate resample through a fixed 512x512.

The first three raised lower-face sharpness from 62% to 94% of the original (the metric includes the added skin texture; the lips themselves score lower). The last two gave only a small improvement, at unchanged speed.

**Why it is not feasible**:

- **The mouth is visibly soft.** The model outputs 256x256 and the face in the test material is about 460x560, roughly a 2x enlargement. Lip-edge and tooth detail is simply absent from the model output and post-processing cannot restore it. After all five steps the mouth is still soft.
- **Too slow on a 3060.** About 21 fps when scaled, and that is lip sync running alone; it is slower still when sharing the GPU with speech synthesis.

## Method 2: SoulX-FlashHead Lite

**How it works**: a diffusion model that generates the whole head, including head motion, expression and blinking, from one portrait image and audio. Output is 512x512 at 25 fps, 24 frames per chunk, 4 sampling steps.

**Measurements** (this model alone, without the language model or speech synthesis; input was the first frame of the test video and 16 seconds of cloned speech; 2 runs):

| Metric | Result |
|---|---|
| Generation speed | about 0.53 s per 24 frames, about 45 fps |
| Scaled to 3060 | about 22 fps, below 25 fps |
| Peak GPU memory | 6.2 to 6.5 GB |
| Compilation at startup | about 7 min 40 s on first launch, about 1 min 36 s on the next |
| Output | 512x512 head video |

Lip contours and teeth are clearer than with MuseTalk and the identity stays consistent. Skin becomes progressively smoother during generation, with an airbrushed look.

**Why it is not feasible**:

- **It cannot take video as input.** It accepts a single image and outputs a cropped 512x512 head, with no step that pastes the result back into a full frame. The body, gestures and background of the streamer's video cannot be used, which conflicts with the premise of using the streamer's own video.
- **No headroom in 12 GB of GPU memory.** It uses 6.2 to 6.5 GB alone; with about 2 GB for speech synthesis and about 3 GB for a local language model the total is about 11.5 GB. The three were not run together. With MuseTalk, once memory overflowed, speed dropped by a factor of tens.
- **Too slow on a 3060.** About 22 fps when scaled and running alone, slower when sharing the GPU with speech synthesis.
- **Slower responses.** It generates in 24-frame chunks, so about one second of audio must accumulate before the first chunk can be produced.
- **Licence restrictions.** The VAE used by the Lite model is under Open-RAIL-M, which carries use restrictions.

It does not install on Windows as published: the multi-GPU library it imports is Linux-only. The tests used single-GPU stand-in modules in `scripts/flashhead_stubs` and left its code unmodified.

## Comparison

| | MuseTalk 1.5 | SoulX-FlashHead Lite |
|---|---|---|
| Input | Video of a real person | Single image |
| Output | Mouth pasted into the original video, same resolution | 512x512 head |
| Model output resolution | 256x256 | 512x512 |
| Speed on 3080 Ti (running alone) | about 43 fps | about 45 fps |
| Scaled to 3060 | about 21 fps | about 22 fps |
| GPU memory (running alone) | about 3.3 GB | 6.2 to 6.5 GB |
| Mouth sharpness | Soft | Clear |
| Blocked by | Picture quality, speed on 3060 | Input form, GPU memory, speed on 3060 |

## Conclusion

Neither method is feasible:

- **MuseTalk** works with the streamer's video, runs in real time and fits in memory, but the mouth does not reach "cannot tell it is synthesized". That is set by the model's resolution, and post-processing has been taken as far as it goes.
- **SoulX-FlashHead Lite** gives a clear mouth, but only generates a head from a single image, so the streamer's video cannot be used, and 12 GB of GPU memory leaves no headroom.
- Both fall below 25 fps when scaled to a 3060.

None of the methods tested so far can deliver "real video input, fully local, real time, sharp mouth" on an RTX 3060 12GB.

Another candidate, Ditto, was only read at source level and not run: it accepts video input and pastes the result back, at 512x512, but upstream only reports testing on an A100 and gives no frame-rate figure.

## What does work

These parts are independent of the lip-sync method and their measurements still hold:

- **Language model**: Qwen3-4B (Q4_K_M, llama.cpp), first sentence in 0.1 to 0.2 s.
- **Voice cloning**: GPT-SoVITS v2ProPlus with streaming synthesis, first audio after about 0.6 s.
- **Real-time scheduling**: clock-driven player with incremental speech and lip sync, about 1 s from comment to speech.

## Reproducing

The scripts assume the development server's layout (`C:\ai_live`).

```powershell
# Install environments and models
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1                  # MuseTalk, GPT-SoVITS, language model
powershell -ExecutionPolicy Bypass -File scripts\setup_envs.ps1 -Only flashhead  # SoulX-FlashHead Lite

# Method 1: real-time pipeline (from the project root, with the sovits environment's python)
python -m core.live --danmaku examples\danmaku_ai.txt --persona_file examples\persona_ai_host.txt `
    --avatar <avatar directory preprocessed by MuseTalk> --ref_audio ref.wav --ref_text "text of the reference audio" `
    --out live.mp4 --deterministic --interval 20

# Method 2: one image plus audio
powershell -ExecutionPolicy Bypass -File scripts\bench_flashhead.ps1 -Image face.png -Audio speech16k.wav -Out out.mp4
```

## Contact

- WeChat: iamfat1
- Email: 5025508@qq.com

## For developers

Technology choices, architecture, memory budget and the measurements from each stage are in [docs/技术方案.md](docs/技术方案.md) (Chinese).

Open-source projects used: Qwen3 (dialogue), GPT-SoVITS (voice), MuseTalk (lip sync), Real-ESRGAN (mouth super-resolution), SoulX-FlashHead (lip sync, tested only), llama.cpp.
