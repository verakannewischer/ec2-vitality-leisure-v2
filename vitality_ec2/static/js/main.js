// Vitality Leisure Park — main.js
// Global utilities

// Animate elements on scroll
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-up');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

document.querySelectorAll('.card, .kpi-card, .chart-container').forEach(el => {
    observer.observe(el);
});
