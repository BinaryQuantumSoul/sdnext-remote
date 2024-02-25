document.addEventListener('DOMContentLoaded', () => {
    setInterval(() => {
        const btn = gradioApp().getElementById('remote_inference_balance_click');
        if (btn) btn.click();
    }, 2000);
});