/**
 * CrowdVision - Professional Crowd Analytics
 * Main JavaScript Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // Initialize Lucide icons
    lucide.createIcons();

    // DOM Elements
    const uploadDropzone = document.getElementById('upload-dropzone');
    const fileInput = document.getElementById('file-input');
    const uploadWrapper = document.getElementById('upload-wrapper');
    const previewWrapper = document.getElementById('preview-wrapper');
    const previewImage = document.getElementById('preview-image');
    const btnRemove = document.getElementById('btn-remove');
    const methodOptions = document.querySelectorAll('.method-option');
    const thresholdWrapper = document.getElementById('threshold-wrapper');
    const thresholdSlider = document.getElementById('threshold-slider');
    const thresholdValue = document.getElementById('threshold-value');
    const btnAnalyze = document.getElementById('btn-analyze');
    const resultsPanel = document.getElementById('results-panel');
    const countNumber = document.getElementById('count-number');
    const resultMethodBadge = document.getElementById('result-method-badge');
    const resultImage = document.getElementById('result-image');
    const btnDownload = document.getElementById('btn-download');
    const btnNew = document.getElementById('btn-new');

    // State
    let selectedFile = null;
    let selectedMethod = 'density';
    let currentResult = null;

    // ============================================
    // Smooth Scroll for Navigation
    // ============================================
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    });

    // ============================================
    // Navbar Scroll Effect
    // ============================================
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const navbar = document.querySelector('.navbar');
        const currentScroll = window.pageYOffset;

        if (currentScroll > 50) {
            navbar.style.boxShadow = '0 4px 20px rgba(0, 0, 0, 0.08)';
        } else {
            navbar.style.boxShadow = 'none';
        }

        lastScroll = currentScroll;
    });

    // ============================================
    // File Upload Handling
    // ============================================
    if (uploadDropzone) {
        uploadDropzone.addEventListener('click', () => fileInput.click());

        uploadDropzone.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadDropzone.classList.add('dragover');
        });

        uploadDropzone.addEventListener('dragleave', () => {
            uploadDropzone.classList.remove('dragover');
        });

        uploadDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadDropzone.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                handleFileSelect(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                handleFileSelect(e.target.files[0]);
            }
        });
    }

    function handleFileSelect(file) {
        // Validate file type
        if (!file.type.startsWith('image/')) {
            showNotification('Please select a valid image file (JPEG, PNG, WEBP)', 'error');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            showNotification('File size must be less than 10MB', 'error');
            return;
        }

        selectedFile = file;

        // Show preview
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImage.src = e.target.result;
            uploadWrapper.querySelector('.upload-dropzone').style.display = 'none';
            previewWrapper.style.display = 'block';
            btnAnalyze.disabled = false;

            // Re-init icons in preview
            lucide.createIcons();
        };
        reader.readAsDataURL(file);
    }

    if (btnRemove) {
        btnRemove.addEventListener('click', () => {
            resetUpload();
        });
    }

    function resetUpload() {
        selectedFile = null;
        fileInput.value = '';
        previewWrapper.style.display = 'none';
        uploadWrapper.querySelector('.upload-dropzone').style.display = 'block';
        uploadWrapper.querySelector('.samples-section').style.display = 'block';
        btnAnalyze.disabled = true;
        resultsPanel.style.display = 'none';

        // Remove selected state from samples
        document.querySelectorAll('.sample-item').forEach(item => {
            item.classList.remove('selected');
        });
    }

    // ============================================
    // Sample Images Gallery
    // ============================================
    const sampleItems = document.querySelectorAll('.sample-item');

    sampleItems.forEach(item => {
        item.addEventListener('click', async () => {
            const sampleUrl = item.dataset.sample;

            // Add selected state
            sampleItems.forEach(s => s.classList.remove('selected'));
            item.classList.add('selected');

            try {
                // Fetch the sample image
                const response = await fetch(sampleUrl);
                const blob = await response.blob();

                // Create a File object
                const filename = sampleUrl.split('/').pop();
                selectedFile = new File([blob], filename, { type: blob.type });

                // Show preview
                previewImage.src = sampleUrl;
                uploadWrapper.querySelector('.upload-dropzone').style.display = 'none';
                uploadWrapper.querySelector('.samples-section').style.display = 'none';
                previewWrapper.style.display = 'block';
                btnAnalyze.disabled = false;

                // Re-init icons
                lucide.createIcons();
            } catch (error) {
                console.error('Error loading sample:', error);
                showNotification('Failed to load sample image', 'error');
            }
        });
    });

    // ============================================
    // Method Selection
    // ============================================
    methodOptions.forEach(option => {
        option.addEventListener('click', () => {
            methodOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            selectedMethod = option.dataset.method;

            // Show/hide threshold for localization
            if (thresholdWrapper) {
                thresholdWrapper.style.display = selectedMethod === 'localization' ? 'block' : 'none';
            }
        });
    });

    if (thresholdSlider) {
        thresholdSlider.addEventListener('input', (e) => {
            thresholdValue.textContent = `${e.target.value}%`;
        });
    }

    // ============================================
    // Analysis
    // ============================================
    if (btnAnalyze) {
        btnAnalyze.addEventListener('click', async () => {
            if (!selectedFile) return;

            // Show loading state
            const btnContent = btnAnalyze.querySelector('.analyze-btn-content');
            const btnLoading = btnAnalyze.querySelector('.analyze-btn-loading');
            btnContent.style.display = 'none';
            btnLoading.style.display = 'flex';
            btnAnalyze.disabled = true;

            try {
                const formData = new FormData();
                formData.append('file', selectedFile);

                let url = `/predict/${selectedMethod}`;
                if (selectedMethod === 'localization') {
                    const threshold = parseInt(thresholdSlider.value) / 100;
                    url += `?threshold=${threshold}`;
                }

                const response = await fetch(url, {
                    method: 'POST',
                    body: formData
                });

                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'Analysis failed');
                }

                const result = await response.json();
                currentResult = result;
                displayResults(result);

            } catch (error) {
                console.error('Analysis error:', error);
                showNotification(`Error: ${error.message}`, 'error');
            } finally {
                btnContent.style.display = 'flex';
                btnLoading.style.display = 'none';
                btnAnalyze.disabled = false;
            }
        });
    }

    // ============================================
    // Display Results
    // ============================================
    function displayResults(result) {
        // Animate count
        animateCount(countNumber, result.count);

        // Update method badge
        resultMethodBadge.textContent = result.method === 'density_map'
            ? 'Density Map'
            : 'Point Detection';

        // Update visualization
        const vizBase64 = result.density_visualization || result.visualization;
        resultImage.src = `data:image/png;base64,${vizBase64}`;

        // Show results panel
        resultsPanel.style.display = 'block';

        // Re-init icons
        lucide.createIcons();

        // Scroll to results
        setTimeout(() => {
            resultsPanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 100);
    }

    function animateCount(element, target) {
        let current = 0;
        const duration = 1000;
        const stepTime = duration / Math.max(target, 30);
        const increment = Math.max(1, Math.ceil(target / 30));

        const timer = setInterval(() => {
            current = Math.min(current + increment, target);
            element.textContent = current;

            if (current >= target) {
                clearInterval(timer);
                element.textContent = target;
            }
        }, stepTime);
    }

    // ============================================
    // Result Actions
    // ============================================
    if (btnDownload) {
        btnDownload.addEventListener('click', () => {
            if (!currentResult) return;

            const vizBase64 = currentResult.density_visualization || currentResult.visualization;
            const link = document.createElement('a');
            link.href = `data:image/png;base64,${vizBase64}`;
            link.download = `crowdvision_result_${currentResult.count}_${Date.now()}.png`;
            link.click();
        });
    }

    if (btnNew) {
        btnNew.addEventListener('click', () => {
            resetUpload();

            // Scroll back to demo section
            const demoSection = document.getElementById('demo');
            if (demoSection) {
                demoSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }
        });
    }

    // ============================================
    // Notification Helper
    // ============================================
    function showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `notification notification-${type}`;
        notification.innerHTML = `
            <span>${message}</span>
            <button onclick="this.parentElement.remove()">
                <i data-lucide="x"></i>
            </button>
        `;

        // Style
        Object.assign(notification.style, {
            position: 'fixed',
            top: '100px',
            right: '20px',
            padding: '1rem 1.5rem',
            background: type === 'error' ? '#fef2f2' : '#f0fdf4',
            border: `1px solid ${type === 'error' ? '#fecaca' : '#bbf7d0'}`,
            borderRadius: '12px',
            boxShadow: '0 10px 40px rgba(0,0,0,0.1)',
            zIndex: '1000',
            display: 'flex',
            alignItems: 'center',
            gap: '1rem',
            animation: 'slideIn 0.3s ease'
        });

        document.body.appendChild(notification);
        lucide.createIcons();

        // Auto remove after 5 seconds
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    // ============================================
    // Intersection Observer for Animations
    // ============================================
    const observerOptions = {
        root: null,
        rootMargin: '0px',
        threshold: 0.1
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
            }
        });
    }, observerOptions);

    // Observe feature cards and step cards
    document.querySelectorAll('.feature-card, .step-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(20px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });
});

// Add slide-in animation
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
`;
document.head.appendChild(style);
