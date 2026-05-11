// ScrollReveal mock to fix 404
export default function scrollReveal() {
  console.log('ScrollReveal initialized');
}

if (typeof document !== 'undefined') {
  document.addEventListener('DOMContentLoaded', () => {
    console.log('ScrollReveal DOM content loaded');
  });
}
