# bkowshik.in

Source for [bkowshik.in](https://bkowshik.in) — my personal site. Built with [Astro](https://astro.build/), deployed to [Cloudflare Pages](https://pages.cloudflare.com/).

## Local development

```bash
npm install
npm run dev     # http://localhost:4321
npm run build   # production build → dist/
```

## Structure

- `src/pages/index.astro` — landing page
- `src/content/pages/` — about, projects, now (markdown)
- `src/content/posts/` — blog posts (markdown / mdx)
- `astro-paper.config.ts` — site config (title, socials, etc.)

## Deployment

Pushes to `main` auto-deploy via Cloudflare Pages.

## Credits

Theme based on [AstroPaper](https://github.com/satnaing/astro-paper) by [Sat Naing](https://satna.ing), MIT licensed.
