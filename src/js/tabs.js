document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-tabs]').forEach((root) => {
    const buttons = root.querySelectorAll('.tab-bar button');
    const panels = root.querySelectorAll('.tab-panel');

    buttons.forEach((button) => {
      button.addEventListener('click', () => {
        const target = button.dataset.tab;

        buttons.forEach((b) => b.classList.toggle('active', b === button));
        panels.forEach((p) => p.classList.toggle('active', p.id === target));
      });
    });
  });
});
