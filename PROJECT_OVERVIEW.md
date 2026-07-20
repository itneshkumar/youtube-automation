This project is a scripted pipeline that turns a single raw screen/webcam recording into an edited, YouTube-ready video, without needing DaVinci Resolve for the main edit.

It takes one input video and automatically removes silence, reduces background noise, and normalizes loudness and EQ so the voice sounds clean and consistent throughout. It transcribes the recording and analyzes the transcript to figure out where an animated motion-graphic explainer would help — for example when the speaker is walking through a list of steps or concepts — and automatically generates and places those animated graphics at the right moments. While a graphic is on screen, the webcam feed is still shown as a small bordered oval bubble overlaid on top, so the presenter stays visible. All of this is stitched back together in the original order into one final rendered video.

Behind the scenes, the recording is split into alternating "talk" segments and "graphic" segments based on timestamps the user (or an LLM) chooses. Each segment is processed on its own and only joined together at the very end, which keeps audio and timing consistent and lets segments be processed in parallel to save time.

The motion graphics themselves are not free-form AI output. An LLM is only used to figure out what content/steps should appear (read from the transcript); the actual animation layout and timing are fixed, hand-built HTML/CSS templates that get rendered to video through a headless browser. This avoids the broken or nonsensical animations that happen when a language model is asked to write animation code directly.

The system can also generate a themed virtual background for the webcam feed, keeping one consistent look across the whole video rather than regenerating it per clip.

There are two ways to run it: a command-line workflow (`start.sh`) that goes from raw recording straight to a finished render, and a simple local browser UI for adjusting settings without editing config files by hand.

In short: record once, and this pipeline automatically cleans the audio, figures out when a visual explainer is needed, builds that visual, overlays the presenter's webcam on top of it, and renders one finished video ready to upload.
