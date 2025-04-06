/* Custom JavaScript for ML Training Framework Documentation */

document.addEventListener('DOMContentLoaded', function() {
    // Add animated entrance effects to feature boxes
    const featureBoxes = document.querySelectorAll('.feature-box');
    if (featureBoxes.length > 0) {
        featureBoxes.forEach((box, index) => {
            box.style.opacity = '0';
            box.style.transform = 'translateY(20px)';
            setTimeout(() => {
                box.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                box.style.opacity = '1';
                box.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    }

    // Create interactive code copy functionality with feedback
    const copyButtons = document.querySelectorAll('.copybtn');
    if (copyButtons.length > 0) {
        copyButtons.forEach(button => {
            const originalTitle = button.getAttribute('title');
            button.addEventListener('click', function() {
                button.textContent = '✓';
                button.style.background = 'var(--success)';
                setTimeout(() => {
                    button.textContent = '';
                    button.style.background = 'var(--accent)';
                }, 2000);
            });
        });
    }

    // Add smooth scroll behavior for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const targetId = this.getAttribute('href');
            if (targetId !== '#') {
                const targetElement = document.querySelector(targetId);
                if (targetElement) {
                    window.scrollTo({
                        top: targetElement.offsetTop - 70,
                        behavior: 'smooth'
                    });
                }
            }
        });
    });

    // Add progress bar at the top of the page when scrolling
    const progressBar = document.createElement('div');
    progressBar.className = 'reading-progress-bar';
    progressBar.style.position = 'fixed';
    progressBar.style.top = '0';
    progressBar.style.left = '0';
    progressBar.style.height = '4px';
    progressBar.style.background = 'var(--accent)';
    progressBar.style.zIndex = '9999';
    progressBar.style.width = '0%';
    progressBar.style.transition = 'width 0.2s ease';
    document.body.appendChild(progressBar);

    window.addEventListener('scroll', function() {
        const scrollTop = window.scrollY;
        const docHeight = document.documentElement.scrollHeight;
        const winHeight = window.innerHeight;
        const scrollPercent = (scrollTop) / (docHeight - winHeight);
        progressBar.style.width = scrollPercent * 100 + '%';
    });

    // Add dark mode toggle listener to apply custom styles
    const darkModeToggle = document.querySelector('.theme-switch-button');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            // Wait a bit for the theme to change then apply additional styles
            setTimeout(() => {
                const isDarkMode = document.body.classList.contains('theme-dark') || 
                                  document.documentElement.dataset.theme === 'dark';
                
                if (isDarkMode) {
                    document.documentElement.style.setProperty('--light', '#252a33');
                } else {
                    document.documentElement.style.setProperty('--light', '#F0F7FF');
                }
            }, 100);
        });
    }

    // Create a "Back to top" button
    const backToTopButton = document.createElement('button');
    backToTopButton.innerHTML = '&uarr;';
    backToTopButton.className = 'back-to-top';
    backToTopButton.style.position = 'fixed';
    backToTopButton.style.bottom = '20px';
    backToTopButton.style.right = '20px';
    backToTopButton.style.borderRadius = '50%';
    backToTopButton.style.width = '50px';
    backToTopButton.style.height = '50px';
    backToTopButton.style.background = 'var(--primary)';
    backToTopButton.style.color = 'white';
    backToTopButton.style.border = 'none';
    backToTopButton.style.cursor = 'pointer';
    backToTopButton.style.opacity = '0';
    backToTopButton.style.transition = 'opacity 0.3s ease';
    backToTopButton.style.zIndex = '998';
    backToTopButton.style.boxShadow = '0 2px 5px rgba(0,0,0,0.2)';
    backToTopButton.style.fontSize = '20px';
    document.body.appendChild(backToTopButton);

    backToTopButton.addEventListener('click', function() {
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });

    window.addEventListener('scroll', function() {
        if (window.scrollY > 300) {
            backToTopButton.style.opacity = '1';
        } else {
            backToTopButton.style.opacity = '0';
        }
    });
});

// Enhance code blocks with line highlighting
window.addEventListener('load', function() {
    const codeBlocks = document.querySelectorAll('pre');
    codeBlocks.forEach(block => {
        const lineNumbers = block.querySelectorAll('.linenos .linenodiv');
        if (lineNumbers.length > 0) {
            const lines = lineNumbers[0].querySelectorAll('span');
            lines.forEach(line => {
                line.addEventListener('mouseenter', function() {
                    const lineNumber = this.textContent;
                    const correspondingLine = block.querySelector(`.highlight .line:nth-child(${lineNumber})`);
                    if (correspondingLine) {
                        correspondingLine.style.background = 'rgba(255, 255, 150, 0.2)';
                    }
                });
                line.addEventListener('mouseleave', function() {
                    const lineNumber = this.textContent;
                    const correspondingLine = block.querySelector(`.highlight .line:nth-child(${lineNumber})`);
                    if (correspondingLine) {
                        correspondingLine.style.background = 'none';
                    }
                });
            });
        }
    });
});