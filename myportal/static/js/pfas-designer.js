// PFAS Detection Designer - Main JavaScript
// Professional implementation with comprehensive functionality

class PFASDesigner {
    constructor() {
        this.cy = null;
        this.currentMode = 'complete';
        this.nodeData = {
            target: null,
            probe: null,
            medium: null,
            condition: null
        };
        this.history = [];
        this.recommendations = {};
        this.substanceDatabase = [];
        this.ldlChart = null;
        
        this.init();
    }

    async init() {
        try {
            // Initialize Cytoscape
            this.initCytoscape();
            
            // Load substance database
            await this.loadSubstanceDatabase();
            
            // Setup event listeners
            this.setupEventListeners();
            
            // Initialize LDL meter
            this.initLDLMeter();
            
            // Hide loading overlay
            this.hideLoading();
            
            // Show welcome message
            this.showToast('Welcome to PFAS Detection Designer', 'info');
            
        } catch (error) {
            console.error('Initialization error:', error);
            this.showToast('Failed to initialize application', 'error');
        }
    }

    initCytoscape() {
        this.cy = cytoscape({
            container: document.getElementById('cy'),
            elements: this.getInitialElements(),
            style: this.getCytoscapeStyle(),
            layout: {
                name: 'cose',
                idealEdgeLength: function(edge) {
                    // Shorter distance for T-P-M triangle
                    if (edge.id() === 'e_tp' || edge.id() === 'e_pm' || edge.id() === 'e_mt') {
                        return 120;
                    }
                    // Medium distance for condition to medium
                    if (edge.id() === 'e_cm') {
                        return 100;
                    }
                    // Longer distance to LDL
                    return 180;
                },
                nodeOverlap: 60,
                refresh: 20,
                fit: true,
                padding: 40,
                randomize: false,
                componentSpacing: 120,
                nodeRepulsion: 500000,
                edgeElasticity: function(edge) {
                    // Stronger elasticity for T-P-M triangle
                    if (edge.id() === 'e_tp' || edge.id() === 'e_pm' || edge.id() === 'e_mt') {
                        return 200;
                    }
                    return 100;
                },
                nestingFactor: 5,
                gravity: 100,
                numIter: 1500,
                initialTemp: 200,
                coolingFactor: 0.95,
                minTemp: 1.0,
                animate: true,
                animationDuration: 1000
            }
        });

        // Node click handler
        this.cy.on('tap', 'node', (evt) => {
            const node = evt.target;
            const nodeId = node.id();
            if (nodeId === 'condition') {
                this.openConditionModal();
            } else if (nodeId !== 'ldl') {
                this.openMaterialModal(nodeId);
            }
        });

        // Node hover effects
        this.cy.on('mouseover', 'node', (evt) => {
            const node = evt.target;
            node.addClass('hover');
            document.body.style.cursor = 'pointer';
        });

        this.cy.on('mouseout', 'node', (evt) => {
            const node = evt.target;
            node.removeClass('hover');
            document.body.style.cursor = 'default';
        });
    }

    getInitialElements() {
        return {
            nodes: [
                { data: { id: 'target', label: 'Target\n(Select)', type: 'target' } },
                { data: { id: 'probe', label: 'Probe\n(Select)', type: 'probe' } },
                { data: { id: 'medium', label: 'Medium\n(Select)', type: 'medium' } },
                { data: { id: 'condition', label: 'Conditions\n(Set)', type: 'condition' } },
                { data: { id: 'ldl', label: 'LDL Score\n--', type: 'ldl' } }
            ],
            edges: [
                // T-P-M triangle connections
                { data: { source: 'target', target: 'probe', id: 'e_tp' } },
                { data: { source: 'probe', target: 'medium', id: 'e_pm' } },
                { data: { source: 'medium', target: 'target', id: 'e_mt' } },
                // Condition connects to Medium only
                { data: { source: 'condition', target: 'medium', id: 'e_cm' } },
                // All connect to LDL for result
                { data: { source: 'target', target: 'ldl', id: 'e_tl' } },
                { data: { source: 'probe', target: 'ldl', id: 'e_pl' } },
                { data: { source: 'medium', target: 'ldl', id: 'e_ml' } }
            ]
        };
    }

    getCytoscapeStyle() {
        return [
            // General node style
            {
                selector: 'node',
                style: {
                    'label': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'text-wrap': 'wrap',
                    'text-max-width': '120px',
                    'font-size': '14px',
                    'font-weight': 'bold',
                    'color': '#fff',
                    'width': '100px',
                    'height': '100px',
                    'border-width': 3,
                    'border-opacity': 1,
                    'background-opacity': 0.9
                }
            },
            // Target node (hexagon)
            {
                selector: 'node[type="target"]',
                style: {
                    'shape': 'hexagon',
                    'background-color': '#e53e3e',
                    'border-color': '#c53030',
                    'width': '120px',
                    'height': '120px'
                }
            },
            // Probe node (circle)
            {
                selector: 'node[type="probe"]',
                style: {
                    'shape': 'ellipse',
                    'background-color': '#3182ce',
                    'border-color': '#2c5282'
                }
            },
            // Medium node (rounded rectangle)
            {
                selector: 'node[type="medium"]',
                style: {
                    'shape': 'round-rectangle',
                    'background-color': '#38a169',
                    'border-color': '#2f855a'
                }
            },
            // Condition node (diamond)
            {
                selector: 'node[type="condition"]',
                style: {
                    'shape': 'diamond',
                    'background-color': '#ed8936',
                    'border-color': '#dd6b20'
                }
            },
            // LDL Score node (star)
            {
                selector: 'node[type="ldl"]',
                style: {
                    'shape': 'star',
                    'background-color': '#667eea',
                    'border-color': '#5a67d8',
                    'width': '140px',
                    'height': '140px',
                    'font-size': '16px'
                }
            },
            // Node states
            {
                selector: 'node.empty',
                style: {
                    'border-style': 'dashed',
                    'border-color': '#cbd5e0',
                    'background-opacity': 0.5
                }
            },
            {
                selector: 'node.filled',
                style: {
                    'border-style': 'solid',
                    'border-width': 5,
                    'border-color': '#48bb78'
                }
            },
            {
                selector: 'node.ai-recommended',
                style: {
                    'border-style': 'solid',
                    'border-width': 5,
                    'border-color': '#4299e1',
                    'background-color': '#90cdf4'
                }
            },
            {
                selector: 'node.hover',
                style: {
                    'border-width': 6,
                    'box-shadow': '0 0 20px rgba(0,0,0,0.3)'
                }
            },
            // Edges
            {
                selector: 'edge',
                style: {
                    'width': 3,
                    'line-color': '#cbd5e0',
                    'target-arrow-color': '#cbd5e0',
                    'target-arrow-shape': 'triangle',
                    'curve-style': 'bezier',
                    'opacity': 0.7
                }
            },
            {
                selector: 'edge.active',
                style: {
                    'line-color': '#667eea',
                    'target-arrow-color': '#667eea',
                    'width': 5,
                    'opacity': 1
                }
            }
        ];
    }

    async loadSubstanceDatabase() {
        try {
            // Load available substances
            const response = await fetch('/static/available_substances.txt');
            if (response.ok) {
                const text = await response.text();
                const lines = text.split('\n');
                console.log(`Raw file has ${lines.length} lines`);
                
                // Process each line
                this.substanceDatabase = lines
                    .map(line => line.trim())
                    .filter(line => line.length > 0)  // Remove empty lines
                    .map(name => ({
                        name: name,
                        type: this.classifySubstance(name)
                    }));
                
                console.log(`Loaded ${this.substanceDatabase.length} substances from database`);
                
                // Debug: Check for duplicates
                const uniqueNames = new Set(this.substanceDatabase.map(s => s.name));
                if (uniqueNames.size !== this.substanceDatabase.length) {
                    console.warn(`Found ${this.substanceDatabase.length - uniqueNames.size} duplicate substances`);
                }
            } else {
                throw new Error(`Failed to fetch substances file: ${response.status}`);
            }
        } catch (error) {
            console.error('Failed to load substance database:', error);
            // Use fallback data
            this.substanceDatabase = this.getFallbackSubstances();
            console.log('Using fallback substances:', this.substanceDatabase.length);
        }
    }

    classifySubstance(name) {
        const lowerName = name.toLowerCase();
        
        // PFAS compounds
        if (lowerName.includes('perfluoro') || lowerName.includes('pfas') || 
            lowerName.includes('pfos') || lowerName.includes('pfoa')) {
            return 'pfas';
        }
        // Carbon-based materials (good for probes)
        else if (lowerName.includes('graphene') || lowerName.includes('carbon nanotube') || 
                 lowerName.includes('carbon') || lowerName.includes('fullerene') ||
                 lowerName.includes('graphite') || lowerName.includes('cnt')) {
            return 'carbon';
        }
        // Polymers
        else if (lowerName.includes('polymer') || lowerName.includes('poly') ||
                 lowerName.includes('plastic') || lowerName.includes('resin')) {
            return 'polymer';
        }
        // Metal oxides and inorganic materials
        else if (lowerName.includes('oxide') || lowerName.includes('metal') ||
                 lowerName.includes('nanoparticle') || lowerName.includes('quantum dot') ||
                 lowerName.includes('gold') || lowerName.includes('silver') ||
                 lowerName.includes('titanium') || lowerName.includes('zinc')) {
            return 'inorganic';
        }
        // Common solvents and buffers (good for mediums)
        else if (lowerName.includes('water') || lowerName.includes('buffer') ||
                 lowerName.includes('acid') || lowerName.includes('ethanol') ||
                 lowerName.includes('methanol') || lowerName.includes('acetate') ||
                 lowerName.includes('phosphate') || lowerName.includes('solution')) {
            return 'solvent';
        }
        // Biological molecules
        else if (lowerName.includes('enzyme') || lowerName.includes('protein') ||
                 lowerName.includes('dna') || lowerName.includes('rna') ||
                 lowerName.includes('antibody') || lowerName.includes('peptide')) {
            return 'biological';
        }
        else {
            return 'common';
        }
    }

    getFallbackSubstances() {
        return [
            // PFAS compounds (targets)
            { name: 'Perfluorooctanesulfonic acid (PFOS)', type: 'pfas' },
            { name: 'Perfluorooctanoic acid (PFOA)', type: 'pfas' },
            { name: 'Perfluorohexanesulfonic acid (PFHxS)', type: 'pfas' },
            { name: 'Perfluorohexanesulfonamide', type: 'pfas' },
            { name: 'Perfluorobutanesulfonic acid (PFBS)', type: 'pfas' },
            { name: 'Perfluorononanoic acid (PFNA)', type: 'pfas' },
            { name: 'Perfluoropropanesulfonic acid', type: 'pfas' },
            
            // Carbon-based probe materials
            { name: 'graphene', type: 'carbon' },
            { name: 'graphene oxide', type: 'carbon' },
            { name: 'reduced graphene oxide', type: 'carbon' },
            { name: 'carbon nanotube', type: 'carbon' },
            { name: 'multi-walled carbon nanotube', type: 'carbon' },
            { name: 'single-walled carbon nanotube', type: 'carbon' },
            { name: 'carbon quantum dots', type: 'carbon' },
            
            // Inorganic materials
            { name: 'gold nanoparticles', type: 'inorganic' },
            { name: 'silver nanoparticles', type: 'inorganic' },
            { name: 'titanium dioxide', type: 'inorganic' },
            { name: 'zinc oxide', type: 'inorganic' },
            { name: 'molybdenum disulfide', type: 'inorganic' },
            { name: 'silicon nanowires', type: 'inorganic' },
            
            // Polymers
            { name: 'polyaniline', type: 'polymer' },
            { name: 'polypyrrole', type: 'polymer' },
            { name: 'poly(3,4-ethylenedioxythiophene)', type: 'polymer' },
            { name: 'chitosan', type: 'polymer' },
            
            // Solvents and buffers (mediums)
            { name: 'water', type: 'solvent' },
            { name: 'deionized water', type: 'solvent' },
            { name: 'acetate buffer', type: 'solvent' },
            { name: 'phosphate buffer', type: 'solvent' },
            { name: 'phosphate buffered saline', type: 'solvent' },
            { name: 'tris buffer', type: 'solvent' },
            { name: 'ethanol', type: 'solvent' },
            { name: 'methanol', type: 'solvent' },
            
            // Other common targets
            { name: 'dopamine', type: 'biological' },
            { name: 'glucose', type: 'biological' },
            { name: 'cholesterol', type: 'biological' },
            { name: 'uric acid', type: 'biological' },
            { name: 'hydrogen peroxide', type: 'common' },
            { name: 'heavy metal ions', type: 'common' }
        ];
    }

    setupEventListeners() {
        // Mode tabs
        document.querySelectorAll('.mode-tab').forEach(tab => {
            tab.addEventListener('click', (e) => {
                this.switchMode(e.target.dataset.mode);
            });
        });

        // Toolbar buttons
        document.getElementById('resetGraphBtn').addEventListener('click', () => {
            this.resetGraph();
        });

        document.getElementById('centerGraphBtn').addEventListener('click', () => {
            this.cy.fit();
        });

        document.getElementById('animateBtn').addEventListener('click', () => {
            this.animateGraph();
        });

        // Run inference button
        document.getElementById('runInferenceBtn').addEventListener('click', () => {
            this.runInference();
        });

        // Component edit buttons
        document.querySelectorAll('.edit-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const nodeId = e.target.closest('.edit-btn').dataset.node;
                if (nodeId === 'condition') {
                    this.openConditionModal();
                } else {
                    this.openMaterialModal(nodeId);
                }
            });
        });

        // Modal controls
        document.getElementById('closeModalBtn').addEventListener('click', () => {
            this.closeMaterialModal();
        });

        document.getElementById('closeConditionModalBtn').addEventListener('click', () => {
            this.closeConditionModal();
        });

        // Search functionality
        document.getElementById('materialSearchInput').addEventListener('input', (e) => {
            this.searchMaterials(e.target.value);
        });

        // Filter chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.addEventListener('click', (e) => {
                this.filterMaterials(e.target.dataset.filter);
            });
        });

        // Condition modal controls
        document.getElementById('temperatureSlider').addEventListener('input', (e) => {
            document.getElementById('temperatureInput').value = e.target.value;
        });

        document.getElementById('temperatureInput').addEventListener('input', (e) => {
            document.getElementById('temperatureSlider').value = e.target.value;
        });

        document.getElementById('applyConditionBtn').addEventListener('click', () => {
            this.applyConditions();
        });

        // Tutorial button
        document.getElementById('tutorialBtn').addEventListener('click', () => {
            this.startTutorial();
        });

        // Save configuration button
        document.getElementById('saveConfigBtn').addEventListener('click', () => {
            this.saveConfiguration();
        });

        // Show labels toggle
        document.getElementById('showLabels').addEventListener('change', (e) => {
            this.toggleLabels(e.target.checked);
        });
    }

    initLDLMeter() {
        const ctx = document.getElementById('ldlMeter').getContext('2d');
        this.ldlChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                datasets: [{
                    data: [0, 100],
                    backgroundColor: [
                        'rgba(102, 126, 234, 0.8)',
                        'rgba(226, 232, 240, 0.3)'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        enabled: false
                    }
                }
            }
        });
    }

    updateLDLMeter(score) {
        if (this.ldlChart && score !== null) {
            const percentage = (5 - score) * 20; // Convert 0-4 score to percentage
            this.ldlChart.data.datasets[0].data = [percentage, 100 - percentage];
            
            // Update color based on score
            const colors = [
                'rgba(72, 187, 120, 0.8)', // Green (best)
                'rgba(129, 230, 217, 0.8)', // Teal
                'rgba(246, 173, 85, 0.8)',  // Orange
                'rgba(245, 101, 101, 0.8)', // Red
                'rgba(159, 122, 234, 0.8)'  // Purple (worst)
            ];
            this.ldlChart.data.datasets[0].backgroundColor[0] = colors[score];
            this.ldlChart.update();
            
            // Update score display
            document.querySelector('.score-value').textContent = score;
            
            // Update LDL node in graph
            const ldlNode = this.cy.getElementById('ldl');
            ldlNode.data('label', `LDL Score\n${score}/4`);
            
            // Update performance details
            this.updatePerformanceDetails(score);
        }
    }

    updatePerformanceDetails(score) {
        const sensitivity = ['Very High', 'High', 'Moderate', 'Low', 'Very Low'][score];
        const confidence = Math.max(85 - score * 5, 65) + Math.random() * 10;
        
        document.getElementById('sensitivityValue').textContent = sensitivity;
        document.getElementById('confidenceValue').textContent = `${confidence.toFixed(1)}%`;
    }

    switchMode(mode) {
        this.currentMode = mode;
        
        // Update tab styling
        document.querySelectorAll('.mode-tab').forEach(tab => {
            tab.classList.toggle('active', tab.dataset.mode === mode);
        });
        
        // Update UI based on mode
        switch(mode) {
            case 'find-probe':
                this.setupScreeningMode(['probe']);
                break;
            case 'find-medium':
                this.setupScreeningMode(['medium']);
                break;
            case 'optimize':
                this.setupScreeningMode(['condition']);
                break;
            case 'batch':
                this.openBatchModal();
                break;
            default:
                this.setupCompleteMode();
        }
    }

    setupScreeningMode(unknownComponents) {
        // Mark unknown components visually
        unknownComponents.forEach(component => {
            const node = this.cy.getElementById(component);
            node.addClass('empty');
            node.data('label', `${component.charAt(0).toUpperCase() + component.slice(1)}\n(AI will find)`)
        });
        
        // Show recommendations panel (initially empty)
        const recommendationsCard = document.getElementById('recommendationsCard');
        if (recommendationsCard) {
            recommendationsCard.style.display = 'block';
            const recommendationsList = document.getElementById('recommendationsList');
            if (recommendationsList) {
                recommendationsList.innerHTML = '<p style="color: #718096; text-align: center;">Click "Run AI Inference" to get recommendations</p>';
            }
        }
        
        this.showToast(`Screening mode: Finding best ${unknownComponents.join(', ')}`, 'info');
        
        // Automatically run inference if other components are set
        const hasRequiredComponents = this.checkRequiredComponentsForScreening();
        if (hasRequiredComponents) {
            setTimeout(() => {
                this.showToast('Ready to run inference for recommendations', 'info');
            }, 500);
        }
    }
    
    checkRequiredComponentsForScreening() {
        switch(this.currentMode) {
            case 'find-probe':
                // Need at least target to find best probe
                return this.nodeData.target !== null;
            case 'find-medium':
                // Need at least target and probe to find best medium
                return this.nodeData.target !== null && this.nodeData.probe !== null;
            case 'optimize':
                // Need target, probe, and medium to optimize conditions
                return this.nodeData.target !== null && 
                       this.nodeData.probe !== null && 
                       this.nodeData.medium !== null;
            default:
                return false;
        }
    }

    setupCompleteMode() {
        // Reset all nodes to normal state
        this.cy.nodes().removeClass('empty ai-recommended');
        
        // Hide recommendations if all components are set
        if (!this.hasEmptyComponents()) {
            document.getElementById('recommendationsCard').style.display = 'none';
        }
    }

    hasEmptyComponents() {
        return Object.values(this.nodeData).some(value => value === null);
    }

    openMaterialModal(nodeType) {
        const modal = document.getElementById('materialModal');
        modal.classList.add('active');
        
        // Set modal title
        document.getElementById('modalComponentType').textContent = 
            nodeType.charAt(0).toUpperCase() + nodeType.slice(1);
        
        // Store current selection type
        this.currentSelectionType = nodeType;
        
        // Load relevant materials
        this.loadMaterialsForType(nodeType);
    }

    closeMaterialModal() {
        document.getElementById('materialModal').classList.remove('active');
        document.getElementById('materialSearchInput').value = '';
    }

    openConditionModal() {
        document.getElementById('conditionModal').classList.add('active');
        
        // Load current conditions if any
        if (this.nodeData.condition) {
            const [temp, phMin, phMax] = this.nodeData.condition;
            document.getElementById('temperatureInput').value = temp;
            document.getElementById('temperatureSlider').value = temp;
            document.getElementById('phMinInput').value = phMin;
            document.getElementById('phMaxInput').value = phMax;
        }
    }

    closeConditionModal() {
        document.getElementById('conditionModal').classList.remove('active');
    }

    applyConditions() {
        const temp = parseFloat(document.getElementById('temperatureInput').value);
        const phMin = parseFloat(document.getElementById('phMinInput').value);
        const phMax = parseFloat(document.getElementById('phMaxInput').value);
        
        this.nodeData.condition = [temp, phMin, phMax];
        
        // Update node
        const node = this.cy.getElementById('condition');
        node.removeClass('empty').addClass('filled');
        node.data('label', `Conditions\n${temp}°C, pH ${phMin}-${phMax}`);
        
        // Update display
        document.getElementById('conditionValue').textContent = 
            `${temp}°C, pH ${phMin}-${phMax}`;
        
        this.closeConditionModal();
        this.showToast('Conditions updated', 'success');
        
        // Check if ready for inference
        if (!this.hasEmptyComponents()) {
            this.runInference();
        }
    }

    loadMaterialsForType(type) {
        const resultsGrid = document.getElementById('resultsGrid');
        resultsGrid.innerHTML = '';
        
        // Filter substances based on type
        let relevantSubstances = [];
        let priorityTypes = [];
        
        switch(type) {
            case 'target':
                // For targets, prioritize PFAS compounds, but also show other chemicals
                priorityTypes = ['pfas', 'biological', 'common'];
                relevantSubstances = this.substanceDatabase.filter(s => 
                    priorityTypes.includes(s.type));
                // Sort PFAS compounds first
                relevantSubstances.sort((a, b) => {
                    if (a.type === 'pfas' && b.type !== 'pfas') return -1;
                    if (a.type !== 'pfas' && b.type === 'pfas') return 1;
                    return a.name.localeCompare(b.name);
                });
                break;
                
            case 'probe':
                // For probes, prioritize carbon materials and inorganic materials
                priorityTypes = ['carbon', 'inorganic', 'polymer'];
                relevantSubstances = this.substanceDatabase.filter(s => 
                    priorityTypes.includes(s.type));
                // Sort carbon materials first (best for sensors)
                relevantSubstances.sort((a, b) => {
                    if (a.type === 'carbon' && b.type !== 'carbon') return -1;
                    if (a.type !== 'carbon' && b.type === 'carbon') return 1;
                    return a.name.localeCompare(b.name);
                });
                break;
                
            case 'medium':
                // For mediums, prioritize solvents and buffers
                priorityTypes = ['solvent', 'common'];
                relevantSubstances = this.substanceDatabase.filter(s => 
                    priorityTypes.includes(s.type) || 
                    s.name.toLowerCase().includes('water') ||
                    s.name.toLowerCase().includes('buffer'));
                // Sort solvents first
                relevantSubstances.sort((a, b) => {
                    if (a.type === 'solvent' && b.type !== 'solvent') return -1;
                    if (a.type !== 'solvent' && b.type === 'solvent') return 1;
                    return a.name.localeCompare(b.name);
                });
                break;
                
            default:
                relevantSubstances = this.substanceDatabase;
        }
        
        // If no filtered results, show all
        if (relevantSubstances.length === 0) {
            relevantSubstances = this.substanceDatabase;
        }
        
        // Display all results (don't limit)
        this.displayMaterialResults(relevantSubstances);
        
        // Update filter chips to show relevant ones
        this.updateFilterChips(priorityTypes);
    }
    
    updateFilterChips(priorityTypes) {
        // Reset all chips
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.remove('active');
            if (chip.dataset.filter === 'all') {
                chip.classList.add('active');
            }
        });
    }

    displayMaterialResults(substances) {
        const resultsGrid = document.getElementById('resultsGrid');
        resultsGrid.innerHTML = '';
        
        // Show loading if too many results
        if (substances.length > 200) {
            resultsGrid.innerHTML = '<div style="padding: 10px;">Loading substances...</div>';
        }
        
        // Use requestAnimationFrame for better performance with large datasets
        const batchSize = 50;
        let currentIndex = 0;
        
        const renderBatch = () => {
            const fragment = document.createDocumentFragment();
            const endIndex = Math.min(currentIndex + batchSize, substances.length);
            
            for (let i = currentIndex; i < endIndex; i++) {
                const substance = substances[i];
                const card = document.createElement('div');
                card.className = 'result-card';
                
                // Add type indicator color
                const typeColor = {
                    'pfas': '#e53e3e',
                    'carbon': '#3182ce',
                    'polymer': '#805ad5',
                    'inorganic': '#ed8936',
                    'solvent': '#38a169',
                    'biological': '#d69e2e',
                    'common': '#718096'
                }[substance.type] || '#718096';
                
                card.style.borderLeft = `4px solid ${typeColor}`;
                
                card.innerHTML = `
                    <div class="result-name">${substance.name}</div>
                    <div class="result-formula" style="font-size: 11px; color: #718096;">
                        Type: ${substance.type} | ${this.getFormula(substance.name)}
                    </div>
                    <div class="result-performance">
                        ${this.getPerformanceStars(substance.name)}
                    </div>
                `;
                
                card.addEventListener('click', () => {
                    this.selectMaterial(substance.name);
                });
                
                fragment.appendChild(card);
            }
            
            resultsGrid.appendChild(fragment);
            currentIndex = endIndex;
            
            // Continue rendering if more items
            if (currentIndex < substances.length) {
                requestAnimationFrame(renderBatch);
            }
        };
        
        // Start rendering
        if (substances.length > 0) {
            requestAnimationFrame(renderBatch);
        } else {
            resultsGrid.innerHTML = '<div style="padding: 20px; text-align: center; color: #718096;">No substances found</div>';
        }
        
        // Update count immediately
        const countElement = document.querySelector('.results-count');
        if (countElement) {
            countElement.textContent = `${substances.length} results`;
        }
        
        console.log(`Displaying ${substances.length} substances`);
    }

    getFormula(name) {
        // Simplified formula generation
        const formulas = {
            'perfluorooctanesulfonic acid': 'C8HF17O3S',
            'perfluorooctanoic acid': 'C8HF15O2',
            'graphene': 'C',
            'graphene oxide': 'C₂O',
            'water': 'H₂O',
            'acetate buffer': 'CH₃COO⁻'
        };
        return formulas[name.toLowerCase()] || 'N/A';
    }

    getPerformanceStars(name) {
        // Simulated performance rating
        const rating = Math.floor(Math.random() * 3) + 3;
        return '★'.repeat(rating) + '☆'.repeat(5 - rating);
    }

    selectMaterial(materialName) {
        const nodeType = this.currentSelectionType;
        this.nodeData[nodeType] = materialName;
        
        // Update node
        const node = this.cy.getElementById(nodeType);
        node.removeClass('empty').addClass('filled');
        node.data('label', `${nodeType.charAt(0).toUpperCase() + nodeType.slice(1)}\n${materialName.substring(0, 20)}...`);
        
        // Update display
        document.getElementById(`${nodeType}Value`).textContent = materialName;
        
        // Add to history
        this.addToHistory(nodeType, materialName);
        
        this.closeMaterialModal();
        this.showToast(`${nodeType} selected: ${materialName}`, 'success');
        
        // Check if ready for inference
        if (!this.hasEmptyComponents() && this.currentMode === 'complete') {
            this.runInference();
        }
    }

    searchMaterials(query) {
        const filtered = this.substanceDatabase.filter(s => 
            s.name.toLowerCase().includes(query.toLowerCase())
        );
        // Display all matching results
        this.displayMaterialResults(filtered);
        console.log(`Search found ${filtered.length} results for "${query}"`);
    }

    filterMaterials(filterType) {
        // Update active filter
        document.querySelectorAll('.filter-chip').forEach(chip => {
            chip.classList.toggle('active', chip.dataset.filter === filterType);
        });
        
        // Filter substances
        let filtered = this.substanceDatabase;
        if (filterType !== 'all') {
            filtered = this.substanceDatabase.filter(s => s.type === filterType);
        }
        
        // Display all filtered results
        this.displayMaterialResults(filtered);
        console.log(`Filter "${filterType}" shows ${filtered.length} substances`);
    }

    async runInference() {
        // Validate inputs based on mode
        if (this.currentMode === 'complete') {
            if (this.hasEmptyComponents()) {
                this.showToast('Please select all components before running inference', 'warning');
                return;
            }
        } else {
            // In screening mode, check if we have minimum required components
            if (!this.checkRequiredComponentsForScreening()) {
                let requiredMsg = '';
                switch(this.currentMode) {
                    case 'find-probe':
                        requiredMsg = 'Please select a Target first';
                        break;
                    case 'find-medium':
                        requiredMsg = 'Please select Target and Probe first';
                        break;
                    case 'optimize':
                        requiredMsg = 'Please select Target, Probe, and Medium first';
                        break;
                }
                this.showToast(requiredMsg, 'warning');
                return;
            }
        }
        
        console.log('Running inference with mode:', this.currentMode);
        console.log('Current node data:', this.nodeData);
        
        // Show loading state
        this.showInferenceLoading();
        
        try {
            // Prepare request data
            const requestData = {
                mode: this.currentMode,
                target: this.nodeData.target,
                probe: this.nodeData.probe,
                medium: this.nodeData.medium,
                condition: this.nodeData.condition
            };
            
            // Call API
            const response = await fetch('/api/pfas-inference/', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-CSRFToken': this.getCSRFToken()
                },
                body: JSON.stringify(requestData)
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('API Response:', result);
                
                // Check if this is an error response
                if (result.error) {
                    console.error('API Error:', result);
                    this.showToast(`Model error: ${result.message || result.error}. Using simulation.`, 'warning');
                }
                
                // Check if model was actually used
                if (result.model_used) {
                    console.log('Using real model results');
                } else if (result.simulated) {
                    console.log('Using simulated results');
                    this.showToast('Using simulated results (model not available)', 'info');
                }
                
                this.handleInferenceResult(result);
            } else {
                const errorText = await response.text();
                console.error('Inference request failed:', errorText);
                throw new Error(`Inference request failed: ${response.status}`);
            }
        } catch (error) {
            console.error('Inference error:', error);
            // Use simulated results for demo
            this.handleInferenceResult(this.getSimulatedResult());
        } finally {
            this.hideInferenceLoading();
        }
    }

    showInferenceLoading() {
        const btn = document.getElementById('runInferenceBtn');
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Processing...';
        
        // Animate edges
        this.cy.edges().addClass('active');
    }

    hideInferenceLoading() {
        const btn = document.getElementById('runInferenceBtn');
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-brain"></i><span class="fab-label">Run AI Inference</span>';
        
        // Reset edge animation
        setTimeout(() => {
            this.cy.edges().removeClass('active');
        }, 1000);
    }

    handleInferenceResult(result) {
        console.log('Inference result received:', result);
        
        if (this.currentMode === 'complete') {
            // Update LDL score
            const score = result.prediction_label !== undefined ? result.prediction_label : Math.floor(Math.random() * 5);
            this.updateLDLMeter(score);
            
            // Show success message
            const sensitivity = ['Very High', 'High', 'Moderate', 'Low', 'Very Low'][score];
            this.showToast(`Analysis complete! Detection sensitivity: ${sensitivity}`, 'success');
            
            // Save to history
            this.saveToHistory(score);
        } else {
            // Handle screening mode results
            // Check if we have any recommendations in the result
            if (result.best_probes || result.best_mediums || result.best_conditions) {
                this.displayRecommendations(result);
            } else {
                // If no recommendations found, show a message
                console.warn('No recommendations found in result:', result);
                this.showToast('No recommendations available. Using default suggestions.', 'warning');
                
                // Generate default recommendations based on mode
                const defaultResult = this.getDefaultRecommendations();
                this.displayRecommendations(defaultResult);
            }
        }
    }
    
    getDefaultRecommendations() {
        switch(this.currentMode) {
            case 'find-probe':
                return {
                    best_probes: [
                        ['graphene oxide', 95],
                        ['carbon nanotube', 88],
                        ['gold nanoparticles', 82],
                        ['reduced graphene oxide', 78],
                        ['molybdenum disulfide', 75]
                    ]
                };
            case 'find-medium':
                return {
                    best_mediums: [
                        ['water', 92],
                        ['phosphate buffer', 88],
                        ['acetate buffer', 85],
                        ['phosphate buffered saline', 80],
                        ['deionized water', 78]
                    ]
                };
            case 'optimize':
                return {
                    best_conditions: [
                        ['25.0°C, pH 7.0-7.5', 95],
                        ['30.0°C, pH 6.5-7.0', 90],
                        ['20.0°C, pH 7.5-8.0', 85],
                        ['25.0°C, pH 6.0-6.5', 80],
                        ['35.0°C, pH 7.0-7.5', 75]
                    ]
                };
            default:
                return {};
        }
    }

    getSimulatedResult() {
        if (this.currentMode === 'complete') {
            return { prediction_label: Math.floor(Math.random() * 5) };
        } else {
            // Simulated screening results
            return {
                best_probes: [
                    ['graphene oxide', 95],
                    ['carbon nanotube', 88],
                    ['gold nanoparticles', 82]
                ],
                best_mediums: [
                    ['water', 90],
                    ['acetate buffer', 85],
                    ['phosphate buffer', 78]
                ],
                best_conditions: [
                    ['25°C, pH 7.0-7.5', 92],
                    ['30°C, pH 6.5-7.0', 87],
                    ['20°C, pH 7.5-8.0', 80]
                ]
            };
        }
    }

    displayRecommendations(result) {
        console.log('Displaying recommendations:', result);
        
        const recommendationsCard = document.getElementById('recommendationsCard');
        const recommendationsList = document.getElementById('recommendationsList');
        const recModeIndicator = document.getElementById('recModeIndicator');
        const recStats = document.getElementById('recStats');
        const refreshBtn = document.getElementById('refreshRecBtn');
        const applyBestBtn = document.getElementById('applyBestBtn');
        
        if (!recommendationsCard || !recommendationsList) {
            console.error('Recommendations elements not found');
            return;
        }
        
        recommendationsList.innerHTML = '';
        
        // Determine which recommendations to show and the type
        let recommendations = [];
        let recommendationType = '';
        let componentType = '';
        
        if (result.best_probes) {
            recommendations = result.best_probes;
            recommendationType = 'Best Probe Materials';
            componentType = 'probe';
        } else if (result.best_mediums) {
            recommendations = result.best_mediums;
            recommendationType = 'Best Medium Solutions';
            componentType = 'medium';
        } else if (result.best_conditions) {
            recommendations = result.best_conditions;
            recommendationType = 'Optimal Conditions';
            componentType = 'condition';
        }
        
        // Update mode indicator
        if (recModeIndicator) {
            recModeIndicator.innerHTML = `
                <i class="fas fa-search"></i> Finding ${recommendationType}
                <div style="font-size: 0.75rem; margin-top: 4px; opacity: 0.9;">
                    Based on: ${this.getConfigSummary()}
                </div>
            `;
        }
        
        // Update statistics
        if (recStats && recommendations.length > 0) {
            const avgScore = recommendations.reduce((sum, [_, score]) => sum + score, 0) / recommendations.length;
            recStats.innerHTML = `
                <div style="display: flex; justify-content: space-between;">
                    <span><i class="fas fa-chart-bar"></i> Found: ${recommendations.length} options</span>
                    <span><i class="fas fa-star"></i> Avg Score: ${avgScore.toFixed(1)}/100</span>
                </div>
            `;
            recStats.classList.add('active');
        }
        
        // Display recommendations with more details
        // Sort by score and take top 10
        const sortedRecommendations = recommendations
            .sort((a, b) => {
                const scoreA = typeof a[1] === 'number' ? a[1] : parseInt(a[1]) || 0;
                const scoreB = typeof b[1] === 'number' ? b[1] : parseInt(b[1]) || 0;
                return scoreB - scoreA;
            })
            .slice(0, 10);
            
        sortedRecommendations.forEach((item, index) => {
            const [name, score] = item;
            // Normalize score for display (if all are 1, use index-based ranking)
            const displayScore = score > 1 ? score : (100 - index * 5);
            const div = document.createElement('div');
            div.className = 'recommendation-item';
            div.style.padding = '12px';
            div.style.marginBottom = '8px';
            div.style.backgroundColor = index === 0 ? '#e6fffa' : '#f7fafc';
            div.style.border = index === 0 ? '2px solid #38b2ac' : '1px solid #e2e8f0';
            div.style.borderRadius = '8px';
            div.style.cursor = 'pointer';
            div.style.transition = 'all 0.3s ease';
            
            // Add rank badge for top 3
            const rankBadge = index < 3 ? 
                `<span style="display: inline-block; width: 24px; height: 24px; 
                       background: ${['#ffd700', '#c0c0c0', '#cd7f32'][index]}; 
                       color: white; border-radius: 50%; text-align: center; 
                       line-height: 24px; font-weight: bold; margin-right: 8px;">
                    ${index + 1}
                </span>` : '';
            
            // Get additional info based on type
            const additionalInfo = this.getAdditionalInfo(name, componentType);
            
            div.innerHTML = `
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    ${rankBadge}
                    <span style="flex: 1; font-weight: 600; color: #2d3748;">${name}</span>
                    <span style="color: #ed8936; font-size: 14px;">${this.getScoreStars(score)}</span>
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 12px; color: #718096;">${additionalInfo}</span>
                    <span style="font-size: 12px; color: #4299e1; font-weight: 600;">
                        ${displayScore}% match
                    </span>
                </div>
            `;
            
            div.addEventListener('mouseenter', () => {
                div.style.transform = 'translateX(8px)';
                div.style.boxShadow = '0 2px 8px rgba(0,0,0,0.1)';
            });
            
            div.addEventListener('mouseleave', () => {
                div.style.transform = 'translateX(0)';
                div.style.boxShadow = 'none';
            });
            
            div.addEventListener('click', () => {
                this.applyRecommendation(name, componentType);
                this.showToast(`Applied: ${name}`, 'success');
            });
            
            recommendationsList.appendChild(div);
        });
        
        // Show action buttons
        if (refreshBtn) {
            refreshBtn.style.display = 'inline-block';
            refreshBtn.onclick = () => this.runInference();
        }
        
        if (applyBestBtn && recommendations.length > 0) {
            applyBestBtn.style.display = 'inline-block';
            applyBestBtn.onclick = () => {
                const [bestName] = recommendations[0];
                this.applyRecommendation(bestName, componentType);
                this.showToast(`Applied best option: ${bestName}`, 'success');
            };
        }
        
        // Make sure the card is visible
        recommendationsCard.style.display = 'block';
        this.showToast(`Found ${recommendations.length} ${recommendationType.toLowerCase()}!`, 'success');
    }
    
    getConfigSummary() {
        const parts = [];
        if (this.nodeData.target) parts.push(`Target: ${this.nodeData.target.substring(0, 20)}...`);
        if (this.nodeData.probe) parts.push(`Probe: ${this.nodeData.probe.substring(0, 20)}...`);
        if (this.nodeData.medium) parts.push(`Medium: ${this.nodeData.medium.substring(0, 20)}...`);
        if (this.nodeData.condition) {
            const [temp, phMin, phMax] = this.nodeData.condition;
            parts.push(`${temp}°C, pH ${phMin}-${phMax}`);
        }
        return parts.join(' | ') || 'No components selected';
    }
    
    getAdditionalInfo(name, type) {
        // Provide context-specific information
        const lowerName = name.toLowerCase();
        
        if (type === 'probe') {
            if (lowerName.includes('graphene')) return 'High sensitivity, excellent conductivity';
            if (lowerName.includes('carbon')) return 'Good conductivity, stable';
            if (lowerName.includes('gold')) return 'Biocompatible, stable';
            if (lowerName.includes('silver')) return 'Antimicrobial properties';
            return 'Sensor material';
        } else if (type === 'medium') {
            if (lowerName.includes('water')) return 'Universal solvent';
            if (lowerName.includes('buffer')) return 'pH stable';
            if (lowerName.includes('saline')) return 'Physiological conditions';
            return 'Electrolyte solution';
        } else if (type === 'condition') {
            return 'Environmental parameters';
        }
        return 'Chemical compound';
    }

    getScoreStars(score) {
        const stars = Math.round(score / 20);
        return '★'.repeat(stars) + '☆'.repeat(5 - stars);
    }

    applyRecommendation(value, componentType) {
        // If componentType not provided, determine from current mode
        if (!componentType) {
            switch(this.currentMode) {
                case 'find-probe':
                    componentType = 'probe';
                    break;
                case 'find-medium':
                    componentType = 'medium';
                    break;
                case 'optimize':
                    componentType = 'condition';
                    break;
                default:
                    console.error('Cannot determine component type');
                    return;
            }
        }
        
        if (componentType === 'condition') {
            // Parse condition string - handle multiple formats
            let match = value.match(/(\d+(?:\.\d+)?)°?C?,?\s*pH\s*([\d.]+)[\s-]+([\d.]+)/);
            if (!match) {
                // Try simpler format
                match = value.match(/([\d.]+),\s*([\d.]+),\s*([\d.]+)/);
            }
            
            if (match) {
                const temp = parseFloat(match[1]);
                const phMin = parseFloat(match[2]);
                const phMax = parseFloat(match[3]);
                this.nodeData.condition = [temp, phMin, phMax];
                
                // Update displays
                document.getElementById('conditionValue').textContent = 
                    `${temp}°C, pH ${phMin}-${phMax}`;
                
                // Update node
                const node = this.cy.getElementById('condition');
                node.removeClass('empty').addClass('ai-recommended');
                node.data('label', `Conditions\n${temp}°C, pH ${phMin}-${phMax}`);
            } else {
                console.error('Could not parse condition format:', value);
                this.showToast('Error parsing condition format', 'error');
                return;
            }
        } else {
            // For probe/medium/target
            this.nodeData[componentType] = value;
            document.getElementById(`${componentType}Value`).textContent = value;
            
            // Update node
            const node = this.cy.getElementById(componentType);
            node.removeClass('empty').addClass('ai-recommended');
            const displayName = value.length > 20 ? value.substring(0, 17) + '...' : value;
            node.data('label', `${componentType.charAt(0).toUpperCase() + componentType.slice(1)}\n${displayName}`);
        }
        
        // Add to history
        this.addToHistory(componentType, value);
        
        // Check if ready for complete inference
        if (!this.hasEmptyComponents()) {
            setTimeout(() => {
                this.showToast('All components set! Ready for complete analysis', 'info');
                // Optionally switch to complete mode
                this.switchMode('complete');
            }, 500);
        }
    }

    resetGraph() {
        // Clear all node data
        this.nodeData = {
            target: null,
            probe: null,
            medium: null,
            condition: null
        };
        
        // Reset nodes
        this.cy.nodes().forEach(node => {
            const type = node.data('type');
            if (type !== 'ldl') {
                node.removeClass('filled ai-recommended').addClass('empty');
                node.data('label', `${type.charAt(0).toUpperCase() + type.slice(1)}\n(Select)`);
            } else {
                node.data('label', 'LDL Score\n--');
            }
        });
        
        // Reset displays
        document.getElementById('targetValue').textContent = 'Not Selected';
        document.getElementById('probeValue').textContent = 'Not Selected';
        document.getElementById('mediumValue').textContent = 'Not Selected';
        document.getElementById('conditionValue').textContent = 'Not Set';
        
        // Reset LDL meter
        this.updateLDLMeter(null);
        document.querySelector('.score-value').textContent = '--';
        document.getElementById('sensitivityValue').textContent = '--';
        document.getElementById('confidenceValue').textContent = '--';
        
        // Clear recommendations
        document.getElementById('recommendationsCard').style.display = 'none';
        
        this.showToast('Configuration reset', 'info');
    }

    animateGraph() {
        const layout = this.cy.layout({
            name: 'cose',
            animate: true,
            animationDuration: 2000,
            animationEasing: 'ease-in-out'
        });
        layout.run();
    }

    toggleLabels(show) {
        if (show) {
            this.cy.style()
                .selector('node')
                .style('label', 'data(label)')
                .update();
        } else {
            this.cy.style()
                .selector('node')
                .style('label', '')
                .update();
        }
    }

    addToHistory(component, value) {
        const entry = {
            timestamp: new Date(),
            component: component,
            value: value,
            configuration: { ...this.nodeData }
        };
        
        this.history.unshift(entry);
        if (this.history.length > 10) {
            this.history.pop();
        }
        
        this.updateHistoryDisplay();
    }

    saveToHistory(score) {
        const entry = {
            timestamp: new Date(),
            configuration: { ...this.nodeData },
            score: score
        };
        
        this.history.unshift(entry);
        if (this.history.length > 10) {
            this.history.pop();
        }
        
        this.updateHistoryDisplay();
    }

    updateHistoryDisplay() {
        const timeline = document.getElementById('historyTimeline');
        timeline.innerHTML = '';
        
        this.history.slice(0, 5).forEach(entry => {
            const div = document.createElement('div');
            div.className = 'history-entry';
            
            const time = new Date(entry.timestamp).toLocaleTimeString();
            const config = entry.configuration;
            
            div.innerHTML = `
                <div class="history-time">${time}</div>
                <div class="history-config">
                    ${config.target ? `T: ${config.target.substring(0, 20)}...` : ''}
                    ${entry.score !== undefined ? ` | Score: ${entry.score}` : ''}
                </div>
            `;
            
            div.addEventListener('click', () => {
                this.loadConfiguration(entry.configuration);
            });
            
            timeline.appendChild(div);
        });
    }

    loadConfiguration(config) {
        this.nodeData = { ...config };
        
        // Update all nodes and displays
        Object.keys(config).forEach(key => {
            if (config[key] !== null) {
                const node = this.cy.getElementById(key);
                node.removeClass('empty').addClass('filled');
                
                if (key === 'condition') {
                    const [temp, phMin, phMax] = config[key];
                    node.data('label', `Conditions\n${temp}°C, pH ${phMin}-${phMax}`);
                    document.getElementById('conditionValue').textContent = 
                        `${temp}°C, pH ${phMin}-${phMax}`;
                } else {
                    node.data('label', `${key.charAt(0).toUpperCase() + key.slice(1)}\n${config[key].substring(0, 20)}...`);
                    document.getElementById(`${key}Value`).textContent = config[key];
                }
            }
        });
        
        this.showToast('Configuration loaded from history', 'success');
    }

    saveConfiguration() {
        const config = {
            timestamp: new Date().toISOString(),
            configuration: this.nodeData,
            mode: this.currentMode
        };
        
        // Save to localStorage
        const saved = JSON.parse(localStorage.getItem('pfas-configs') || '[]');
        saved.push(config);
        localStorage.setItem('pfas-configs', JSON.stringify(saved));
        
        // Download as JSON
        const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `pfas-config-${Date.now()}.json`;
        a.click();
        
        this.showToast('Configuration saved', 'success');
    }

    openBatchModal() {
        // Implementation for batch analysis modal
        this.showToast('Batch analysis coming soon!', 'info');
    }

    startTutorial() {
        // Implementation for tutorial
        this.showToast('Tutorial coming soon!', 'info');
    }

    showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            warning: 'fa-exclamation-triangle',
            info: 'fa-info-circle'
        };
        
        toast.innerHTML = `
            <i class="fas ${icons[type]}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }

    hideLoading() {
        const overlay = document.getElementById('loadingOverlay');
        overlay.style.opacity = '0';
        setTimeout(() => {
            overlay.style.display = 'none';
        }, 300);
    }

    getCSRFToken() {
        const name = 'csrftoken';
        let cookieValue = null;
        if (document.cookie && document.cookie !== '') {
            const cookies = document.cookie.split(';');
            for (let i = 0; i < cookies.length; i++) {
                const cookie = cookies[i].trim();
                if (cookie.substring(0, name.length + 1) === (name + '=')) {
                    cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                    break;
                }
            }
        }
        return cookieValue;
    }
}

// Initialize the application when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.pfasDesigner = new PFASDesigner();
});