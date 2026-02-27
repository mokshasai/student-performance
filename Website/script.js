document.addEventListener('DOMContentLoaded', () => {
  // ── Active Nav ───────────────────────────────────────────────
  const currentPage = window.location.pathname.split('/').pop() || 'index.html';
  document.querySelectorAll('nav a').forEach(link => {
    if (link.getAttribute('href') === currentPage) link.classList.add('active');
  });

  // ── Fade-in on scroll ────────────────────────────────────────
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('visible');
        e.target.style.opacity = '1';
        e.target.style.transform = 'translateY(0)';
      }
    });
  }, { threshold: 0.08 });

  document.querySelectorAll('.card, .info-tile, figure, .img-wrap, .callout').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(22px)';
    el.style.transition = 'opacity 0.55s ease, transform 0.55s ease';
    observer.observe(el);
  });

  // ── Smooth table row highlight ───────────────────────────────
  document.querySelectorAll('table tbody tr').forEach(row => {
    row.addEventListener('mouseenter', () => row.style.background = 'rgba(99,179,237,0.06)');
    row.addEventListener('mouseleave', () => row.style.background = '');
  });

  // ── Progress bars animate on scroll ─────────────────────────
  const fillObs = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        const fill = e.target;
        const target = fill.getAttribute('data-pct');
        fill.style.width = target + '%';
        fillObs.unobserve(fill);
      }
    });
  }, { threshold: 0.5 });

  document.querySelectorAll('.progress-fill').forEach(el => {
    el.style.width = '0%';
    el.style.transition = 'width 1.2s cubic-bezier(0.4,0,0.2,1)';
    fillObs.observe(el);
  });
});