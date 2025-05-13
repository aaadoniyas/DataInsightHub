document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = document.querySelectorAll('[data-bs-toggle="tooltip"]')
    const tooltipList = [...tooltipTriggerList].map(tooltipTriggerEl => new bootstrap.Tooltip(tooltipTriggerEl))
    
    // Handle tab switching
    const triggerTabList = document.querySelectorAll('#analysis-tabs button')
    triggerTabList.forEach(triggerEl => {
        const tabTrigger = new bootstrap.Tab(triggerEl)
        triggerEl.addEventListener('click', event => {
            event.preventDefault()
            tabTrigger.show()
        })
    })
    
    // Create PCA variance chart
    const varianceCtx = document.getElementById('varianceChart');
    if (varianceCtx) {
        const varianceData = JSON.parse(varianceCtx.getAttribute('data-variance'));
        const cumulativeData = JSON.parse(varianceCtx.getAttribute('data-cumulative'));
        const labels = varianceData.map((_, index) => `PC${index + 1}`);
        
        new Chart(varianceCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Variance Explained (%)',
                        data: varianceData,
                        backgroundColor: 'rgba(54, 162, 235, 0.5)',
                        borderColor: 'rgba(54, 162, 235, 1)',
                        borderWidth: 1
                    },
                    {
                        label: 'Cumulative Variance (%)',
                        data: cumulativeData,
                        type: 'line',
                        fill: false,
                        borderColor: 'rgba(255, 99, 132, 1)',
                        tension: 0.1,
                        pointBackgroundColor: 'rgba(255, 99, 132, 1)'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 100,
                        title: {
                            display: true,
                            text: 'Variance Explained (%)'
                        }
                    },
                    x: {
                        title: {
                            display: true,
                            text: 'Principal Components'
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'PCA Variance Explained'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                label += context.parsed.y.toFixed(2) + '%';
                                return label;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Toggle loading spinner when download button is clicked
    const downloadBtn = document.getElementById('download-btn');
    if (downloadBtn) {
        downloadBtn.addEventListener('click', function() {
            const spinner = this.querySelector('.spinner-border');
            const btnText = this.querySelector('.btn-text');
            
            spinner.classList.remove('d-none');
            btnText.textContent = 'Generating Excel...';
            
            // Reset button after download starts (after 2 seconds)
            setTimeout(() => {
                spinner.classList.add('d-none');
                btnText.textContent = 'Download Excel Report';
            }, 2000);
        });
    }
    
    // Expand/collapse measurement groups
    const toggleButtons = document.querySelectorAll('.toggle-measurements');
    toggleButtons.forEach(button => {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const targetElement = document.getElementById(targetId);
            const icon = this.querySelector('i');
            
            if (targetElement.classList.contains('show')) {
                targetElement.classList.remove('show');
                icon.classList.remove('fa-chevron-up');
                icon.classList.add('fa-chevron-down');
            } else {
                targetElement.classList.add('show');
                icon.classList.remove('fa-chevron-down');
                icon.classList.add('fa-chevron-up');
            }
        });
    });
});
