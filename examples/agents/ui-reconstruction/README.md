# Rebuild a UI shown in a video

**Prompt**

> Watch demo.mp4 and recreate the login page in React.

**Workflow**

1. `context` with the task → relevant ranges, evidence, UI states, frames (image paths)
2. open the frame images for layout and style; use UI-state elements for exact text
3. implement; list anything not observable as an assumption

**Operations**

```bash
videocontent context demo.mp4 "Recreate the login page UI" --agent
```

**Output** (fields abridged)

```text
intent: ui_state_query
evidence:
  00:00:41.340 [observed · transcript] Now the demo, the browser opens the login page on localHistPort3000.
  00:00:41.600 [observed · ocr]        localhost:3000/login
ui_states (derived from stable on-screen text):
  00:00:41.600 → 00:00:48.333  ['localhost:3000/login', 'Sign in', 'email@example.com', 'Password']
frames:
  00:00:41.600  image: …/.videocontent/demo-<hash>/frames/pass/f_00064.jpg
  00:00:47.800  image: …/.videocontent/demo-<hash>/frames/pass/f_00073.jpg
```

Frame `f_00064.jpg` (the image the agent opens) shows the page: a browser bar with
`localhost:3000/login`, a "Sign in" heading, an email field (placeholder
`email@example.com`), a password field, and a blue "Log in" button, left-aligned on a light
gray background.

**What the agent builds from it**

```tsx
// Observed at 00:00:41.6–00:00:48.3 in demo.mp4 (route /login).
export function LoginPage() {
  return (
    <main className="min-h-screen bg-slate-100 px-24 pt-24">
      <h1 className="text-5xl text-slate-900">Sign in</h1>
      <form className="mt-10 flex w-[520px] flex-col gap-8">
        <input type="email" placeholder="email@example.com" className="rounded border border-slate-400 bg-white px-4 py-3 text-xl" />
        <input type="password" placeholder="Password" className="rounded border border-slate-400 bg-white px-4 py-3 text-xl" />
        <button type="submit" className="w-56 rounded bg-blue-500 py-3 text-xl text-white">Log in</button>
      </form>
    </main>
  );
}
```

Assumptions the agent states: exact colors and spacing are estimated from the frame; no
validation, error or hover states were visible in the recording.
