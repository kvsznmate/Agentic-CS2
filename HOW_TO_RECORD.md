# How to Record Gameplay Data for Agentic-CS2 project

This guide walks you through recording gameplay, step by step. **You do not need
to know how to program.** You'll type a few commands exactly as shown, play some
Counter-Strike 2, and press a key to stop. That's it.

Follow the steps in order the first time. Once you've done it once, recording
again is just **Part 5** each time.

---

## Before you start: what you'll be doing

You'll open a small text-based program called a **terminal**, type a few lines to
start the recorder, then play CS2 normally. While you play, the program quietly
saves what's on your screen and which keys and mouse movements you make. When
you're done, you press one key and it saves everything to a folder.

Every command is written out for you to copy exactly.

---

## Part 1  Get the settings right (do this first)

The recorder expects your game and screen to be set up a specific way. **If these
aren't right, the data will be unusable, so don't skip this.**

### 1a. Set your Windows screen resolution to 1920 × 1080

1. Right-click anywhere on your **desktop** (the empty background).
2. Click **Display settings**.
3. Scroll down to **Display resolution**.
4. Click the dropdown and choose **1920 × 1080**.
5. Click **Keep changes** when it asks.

> **If 1920 × 1080 is not in the list, or your screen says another number is
> "Recommended":** stop and tell the developer before recording. The recorder is
> set up for 1920 × 1080; a different resolution needs a one-time adjustment they
> have to make. Don't record until this is sorted — the data won't line up
> otherwise.

### 1b. Set CS2 to the matching settings

Open CS2, go to **Settings → Video → Video** and set:

- **Display Mode:** Fullscreen
- **Resolution:** 1920 × 1080
- **Aspect Ratio:** 16:9
- **Brightness:** 0.93

Then go to **Settings → Video → Advanced video** and set:

- **Boost Player Contrast:** ENABLED
- V-Sync: DISABLED
- **NVidia Reflex Low Latency:** ENABLED

- **Current Video Values Preset:** LOW
- Multisampling Anti-Aliasing Mode: NONE
- **Global Shadow Quality:** LOW
- **Dynamic Shadows:** Sun only

**Everything else either Disabled or LOW.** If you're unsure, ask the developer.

### 1c. Make sure the radar/minimap is set up correctly

In cs2 go to **Settings → GAME → RADAR** and set:

- **Radar Centers The Player:** NO
- **Radar is rotating:** NO
- **Radar Hud Map Blends with background:** YES
- **Blur Background:** YES
- **Radar Hud Background Opacity:** 100%
- **Radar Hud Size:** 1.0
- **Radar Map zoom:** 0.35
- **Radar map alternate Zoom:** 1.0

- **Toggle Square shape with scoreboard:** YES
- **Force Square Shape:** YES
- **Radar is zooming Dynamically:** NO

#### Other settings may differ eg.: crosshair, mouse sensitivity, keybinds. But important to provide the developer with your settings and to use the same settings for all recordings. If you change them, let the developer know.
---

## Part 2 — Open the terminal and go to the project folder

"The terminal" is a window where you type commands. Here's how to open the right
one:

1. Click the **Windows Start** menu (bottom-left).
2. Type **Anaconda Prompt**.
3. Click **Anaconda Prompt** in the results. A black or dark window opens with a
   blinking cursor. This is the terminal.

Now point it at the project folder. **Copy the line below exactly** (you can
copy-paste — right-click in the terminal window usually pastes), then press
**Enter**:

```
cd Desktop\Agentic-CS2
```

> `cd` means "change directory" — it moves the terminal into the project folder.
> If your project is somewhere else, ask the developer for the correct path to
> put after `cd`.

---

## Part 3 — Turn on the environment (do this every time you open a terminal)

The project needs its own set of tools switched on. Type this line and press
**Enter**:

```
conda activate agentic-cs2
```

You'll know it worked because the start of the line in the terminal will now show
**`(agentic-cs2)`** at the front. If it does, you're ready.

> If you ever open a fresh terminal, you have to do **Part 2** (`cd …`) and
> **Part 3** (`conda activate …`) again. They don't stick between windows.

---

## Part 4 — First time only: a quick check that everything works

Do this **once**, before your very first real recording, to make sure the mouse
and screen are being read correctly.

Type this and press **Enter**:

```
python -m src.recorder --verify
```

The program will give you instructions on screen. It will ask you to:

1. **Sweep your mouse steadily to the RIGHT** for a few seconds, then
2. **Sweep your mouse steadily to the LEFT** for a few seconds.

(Do this with CS2 in front, as if you were playing.) When it finishes, it prints
a result:

- **`RESULT: PASS`** → Everything is reading correctly. You're good to record.
- **`RESULT: MOSTLY OK`** → Almost, You're good to record.
  from right to left promptly when it tells you to.
- **`RESULT: FAIL`** → **Retry a few times if still persist** Stop and tell the developer.** Something isn't reading
  right, and recording now would waste your time. Don't continue until this
  passes.

You only need to see PASS once on your setup. After that, you can skip this step
unless something about your computer or game setup changes.

---

## Part 5 — Record your gameplay (this is the main step, repeat as needed)

This is what you'll do each time you want to record. Make sure:

- CS2 is open, in an **match**, fullscreen (Parts 1 & 2 done),
- the terminal shows **`(agentic-cs2)`** at the front of the line (Part 3 done).

Type this and press **Enter**:

```
python -m src.recorder --record
```

The recorder starts. You'll see numbers updating in the terminal (how many frames
it's saved, and so on) — that's normal, you can ignore them.

**Now click back into CS2 and play normally.** Move around, aim, shoot.
Play the way you actually play.


### When you're finished recording

Press the **`F8`** key on your keyboard. (F8 is along the top row of the
keyboard.)

The recorder stops, saves everything, and prints a summary in the terminal,
including the folder it saved to. That's one recording done.

**To record again**, just run the same command (`python -m src.recorder --record`)
and repeat. Each time makes a new, separate recording. **The more sessions you
record, the better** aim to build up several, across different matches of play.

> If the terminal ever says it's **refusing to start
> because the disk is full**, you need to free up space (or ask the developer).

---

## Part 6 — Where your recordings are saved

Everything you record goes into this folder:

```
Desktop\Agentic-CS2\data\recordings\
```

Inside it, **each recording is its own folder**, named with the date and time it
was made  for example:

```
data\recordings\
├── session_20260811_141839\      ← one recording (made 2026-08-11 at 14:18)
├── session_20260811_142126\      ← another recording
└── ...
```

You don't need to open or touch these folders — the developer's tools read them.
Just don't rename or delete them unless asked.

---

## Part 7 — What's inside a recording (for your understanding)

Each recording folder contains, for **every single moment** of your play (many
times per second):

- **A picture of your screen**  what you saw as you played.
- **A close-up of the minimap** the little map from the top-left corner, saved
  in higher detail so your position on the map is readable.
- **What you did that instant** which movement keys were held (W, A, S, D and a
  few others), whether you were clicking (shooting), and how you moved the mouse.

All three are saved together and lined up in time, so that for any moment, the
picture and the action you took are matched up exactly. That pairing — screen and
action, perfectly in step is the whole reason this data is useful for teaching
the game-playing model.

There's also a small `manifest.json` file in each folder — that's just the
recorder's own bookkeeping (how many frames, whether the recording finished
cleanly). You can ignore it.

---

## Quick recap (once you've done it once)

Every time you want to record:

1. Open **Anaconda Prompt**.
2. `cd C:\Users\matek_yulq090\Desktop\Agentic-CS2`
3. `conda activate agentic-cs2`
4. Start CS2 → **match**, fullscreen 1920×1080.
5. `python -m src.recorder --record`
6. Click into CS2 and **play**.
7. Press **`F8`** to stop and save.
8. Recordings appear in `data\recordings\`.

---

## If something goes wrong

- **The terminal doesn't show `(agentic-cs2)`** → you missed Part 3; type
  `conda activate agentic-cs2` and press Enter.
- **"No such file or directory" or the command isn't found** → you're probably
  not in the project folder; redo Part 2 (the `cd …` line).
- **The alignment check (Part 4) says FAIL** → don't record; tell the developer.
- **It refuses to start because the disk is full** → free up space, or ask the
  developer.
- **1920 × 1080 isn't available, or your screen recommends a different
  resolution** → tell the developer before recording.
- **Anything else that looks wrong** → stop, note what the terminal says, and ask
  the developer rather than guessing.
