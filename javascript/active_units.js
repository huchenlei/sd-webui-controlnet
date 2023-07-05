/**
 * Give a badge on ControlNet Accordion indicating total number of active 
 * units.
 * Make active unit's tab name green.
 * Append control type to tab name.
 */
(function () {
    const cnetAllUnits = new Map/* <Element, GradioTab> */();
    const cnetAllAccordions = new Set();
    onUiUpdate(() => {
        function childIndex(element) {
            // Get all child nodes of the parent
            let children = Array.from(element.parentNode.childNodes);

            // Filter out non-element nodes (like text nodes and comments)
            children = children.filter(child => child.nodeType === Node.ELEMENT_NODE);

            return children.indexOf(element);
        }

        function imageInputDisabledAlert() {
            alert('Inpaint control type must use a1111 input in img2img mode.');
        }

        class GradioTab {
            constructor(tab) {
                this.tab = tab;
                this.isImg2Img = tab.querySelector('.cnet-unit-enabled').id.includes('img2img');

                this.enabledCheckbox = tab.querySelector('.cnet-unit-enabled input');
                this.inputImage = tab.querySelector('.cnet-input-image-group .cnet-image input[type="file"]');
                this.controlTypeRadios = tab.querySelectorAll('.controlnet_control_type_filter_group input[type="radio"]');

                const tabs = tab.parentNode;
                this.tabNav = tabs.querySelector('.tab-nav');
                this.tabIndex = childIndex(tab) - 1; // -1 because tab-nav is also at the same level.

                this.attachEnabledButtonListener();
                this.attachControlTypeRadioListener();
                this.attachTabNavChangeObserver();
                this.attachImageUploadListener();
            }

            getTabNavButton() {
                return this.tabNav.querySelector(`:nth-child(${this.tabIndex + 1})`);
            }

            getActiveControlType() {
                for (let radio of this.controlTypeRadios) {
                    if (radio.checked) {
                        return radio.value;
                    }
                }
                return undefined;
            }

            updateActiveState() {
                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                if (this.enabledCheckbox.checked) {
                    tabNavButton.classList.add('cnet-unit-active');
                } else {
                    tabNavButton.classList.remove('cnet-unit-active');
                }
            }

            /**
             * Add the active control type to tab displayed text.
             */
            updateActiveControlType() {
                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                // Remove the control if exists
                const controlTypeSuffix = tabNavButton.querySelector('.control-type-suffix');
                if (controlTypeSuffix) controlTypeSuffix.remove();

                // Add new suffix.
                const controlType = this.getActiveControlType();
                if (controlType === 'All') return;

                const span = document.createElement('span');
                span.innerHTML = `[${controlType}]`;
                span.classList.add('control-type-suffix');
                tabNavButton.appendChild(span);
            }

            /**
             * When 'Inpaint' control type is selected in img2img:
             * - Make image input disabled
             * - Clear existing image input
             */
            updateImageInputState() {
                if (!this.isImg2Img) return;

                const tabNavButton = this.getTabNavButton();
                if (!tabNavButton) return;

                const controlType = this.getActiveControlType();
                if (controlType.toLowerCase() === 'inpaint') {
                    this.inputImage.disabled = true;
                    this.inputImage.parentNode.addEventListener('click', imageInputDisabledAlert);
                    const removeButton = this.tab.querySelector(
                        '.cnet-input-image-group .cnet-image button[aria-label="Remove Image"]');
                    if (removeButton) removeButton.click();
                } else {
                    this.inputImage.disabled = false;
                    this.inputImage.parentNode.removeEventListener('click', imageInputDisabledAlert);
                }
            }

            attachEnabledButtonListener() {
                this.enabledCheckbox.addEventListener('change', () => {
                    this.updateActiveState();
                });
            }

            attachControlTypeRadioListener() {
                for (const radio of this.controlTypeRadios) {
                    radio.addEventListener('change', () => {
                        this.updateActiveControlType();
                        this.updateImageInputState();
                    });
                }
            }

            /**
             * Each time the active tab change, all tab nav buttons are cleared and
             * regenerated by gradio. So we need to reapply the active states on 
             * them.
             */
            attachTabNavChangeObserver() {
                new MutationObserver((mutationsList) => {
                    for (const mutation of mutationsList) {
                        if (mutation.type === 'childList') {
                            this.updateActiveState();
                            this.updateActiveControlType();
                        }
                    }
                }).observe(this.tabNav, { childList: true });
            }

            attachImageUploadListener() {
                // Automatically check `enable` checkbox when image is uploaded.
                this.inputImage.addEventListener('change', (event) => {
                    if (!event.target.files) return;
                    if (!this.enabledCheckbox.checked)
                        this.enabledCheckbox.click();
                });
            }
        }

        gradioApp().querySelectorAll('.cnet-unit-tab').forEach(tab => {
            if (cnetAllUnits.has(tab)) return;
            cnetAllUnits.set(tab, new GradioTab(tab));
        });

        function getActiveUnitCount(checkboxes) {
            let activeUnitCount = 0;
            for (const checkbox of checkboxes) {
                if (checkbox.checked)
                    activeUnitCount++;
            }
            return activeUnitCount;
        }

        gradioApp().querySelectorAll('#controlnet').forEach(accordion => {
            if (cnetAllAccordions.has(accordion)) return;
            const checkboxes = accordion.querySelectorAll('.cnet-unit-enabled input');
            if (!checkboxes) return;

            const span = accordion.querySelector('.label-wrap span');
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('change', () => {
                    // Remove existing badge.
                    if (span.childNodes.length !== 1) {
                        span.removeChild(span.lastChild);
                    }
                    // Add new badge if necessary.
                    const activeUnitCount = getActiveUnitCount(checkboxes);
                    if (activeUnitCount > 0) {
                        const div = document.createElement('div');
                        div.classList.add('cnet-badge');
                        div.classList.add('primary');
                        div.innerHTML = `${activeUnitCount} unit${activeUnitCount > 1 ? 's' : ''}`;
                        span.appendChild(div);
                    }
                });
            });
            cnetAllAccordions.add(accordion);
        });
    });
})();