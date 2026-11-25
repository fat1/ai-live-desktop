[简体中文](README.md) | **English**

# AI Avatar Live Assistant (ai-live-local)

Turn your own video and voice into a talking digital host that runs entirely on your own PC.

Give it a few videos of a real person and the content you want covered. The app makes the person in the video speak that content with matching lip movement and a voice that sounds like them, then feeds the picture and sound into Douyin Live Companion (抖音直播伴侣) so you can go live. Everything runs on your computer: no cloud service, no per-minute fees, and your footage is never uploaded anywhere.

> **This project is under active development and there is no installer to download yet.** Star the repository to follow progress, or contact the author below.

The app's interface is in Chinese and it is built for streaming on Douyin.

## What it does

- **Makes the person in your video talk**: upload a video of a real person and the avatar streams with that appearance, with lips in sync with the speech.
- **Uses your own voice**: record a few seconds of audio and the app learns that voice. You can also pick one of the built-in voices.
- **Answers viewer comments on its own**: tell it about your products and persona and the questions viewers are likely to ask, and it composes the replies.
- **Switches smoothly between several videos**: upload multiple clips to rotate through, with no sudden jumps in the picture.
- **Connects directly to Douyin Live Companion**: no extra streaming software required.

## Go live in four steps

1. **Choose a look**: upload one or more videos of a real person.
2. **Choose a voice**: pick a built-in voice, or upload a few seconds of audio to clone your own.
3. **Add your content**: describe the persona and products, and import the viewer comments or script to respond to. Templates are included.
4. **Go live**: preview the result inside the app, then follow the illustrated guide to connect Douyin Live Companion.

No technical knowledge or parameter tuning is needed. The app checks your computer and picks suitable settings automatically.

## System requirements

| Item | Requirement |
|---|---|
| OS | Windows 10 or Windows 11 (64-bit) |
| Graphics card | NVIDIA discrete GPU with 12 GB of video memory or more, for example RTX 3060 12GB |
| Memory | 16 GB or more, 32 GB recommended |
| Disk | At least 30 GB of free space |

Not sure whether your PC qualifies? The app checks on first launch and tells you.

Laptops, AMD graphics cards and Macs are not supported for now.

## Getting the best results from your footage

- **Video**: face the camera, upper body in frame, even lighting. Do not stand too far away, and keep hands and microphones away from the mouth. Anything from 30 seconds to a few minutes works.
- **Multiple videos**: shoot them in the same place, from the same camera position, in the same outfit. Transitions are then nearly invisible.
- **Voice**: speak at a normal pace for 5 to 10 seconds in a quiet room, with no background music.

## FAQ

**Does it cost anything?**
The software is free and open source. All processing happens on your own computer, so there are no usage-based fees.

**Does it need to stay online?**
Only the first launch needs a connection, to download the required files (about 10 GB). After that the app works offline. Streaming itself still needs a network, of course.

**Are my videos and voice uploaded?**
No. All material stays on your computer.

**Can I use someone else's video or voice?**
Only with that person's consent. Using someone's likeness or voice without permission is illegal.

**Is avatar streaming allowed by the platform?**
Douyin requires AI-generated content to be labelled, and individual categories have their own rules. Check the platform's current requirements before going live.

## Contact

Questions, suggestions, or interest in testing:

- WeChat: iamfat1
- Email: 5025508@qq.com

## For developers

Technology choices, architecture, memory budget and the development plan are in [docs/技术方案.md](docs/技术方案.md) (Chinese).

Built on these open-source projects: Qwen3 (dialogue), GPT-SoVITS (voice), MuseTalk (lip sync), RIFE (video transitions), llama.cpp and PySide6.
