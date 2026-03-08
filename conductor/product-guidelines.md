# Product Guidelines: ymago

## Prose Style
- **Technical & Direct:** Documentation and internal communications should use precise, technical, and objective language. Avoid fluff and focus on providing clear, accurate information that is easy for developers to parse.

## UX Principles
- **Interactive & Visual Feedback:** The CLI must prioritize real-time status updates using spinners, progress bars, and formatted output from the `rich` library. Users should always feel "in control" and informed about the progress of asynchronous operations.

## Branding & Voice
- **Creative & Experimental:** The project's voice should highlight the cutting-edge and experimental nature of generative AI. It should feel modern and forward-thinking, encouraging users to explore the creative possibilities of the supported models.

## Error Handling & Feedback
- **Rich & Formatted Errors:** Use `rich` panels, colors, and formatting to clearly separate error details from standard output, making them easy to identify.
- **Actionable & Helpful Failures:** Error messages must provide direct suggestions or potential fixes to guide the user toward a resolution.
- **Clean Terminal on Error:** Detailed stack traces and debug logs should be directed to log files by default, keeping the terminal output focused on actionable information unless a `--verbose` flag is used.
