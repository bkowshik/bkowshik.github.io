import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://bkowshik.in/",
    title: "Bhargav Kowshik",
    description:
      "Senior data scientist working on noisy real-world sensor data. Moving toward neurotech and biosignal applications.",
    author: "Bhargav Kowshik",
    profile: "https://bkowshik.in/about",
    ogImage: "default-og.jpg",
    lang: "en",
    timezone: "Asia/Kolkata",
    dir: "ltr",
  },
  posts: {
    perPage: 6,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: { enabled: false },
    search: "pagefind",
  },
  socials: [
    { name: "github",   url: "https://github.com/bkowshik" },
    { name: "linkedin", url: "https://www.linkedin.com/in/bkowshik/" },
    { name: "x",        url: "https://x.com/bkowshik" },
    { name: "mail",     url: "mailto:bhargav.kowshik@gmail.com" },
    { name: "rss",      url: "/rss.xml", linkTitle: "RSS feed" },
  ],
  shareLinks: [
    { name: "x",        url: "https://x.com/intent/post?url=" },
    { name: "linkedin", url: "https://www.linkedin.com/sharing/share-offsite/?url=" },
    { name: "mail",     url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
