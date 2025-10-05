# Contributing to Fixed-Site vs Festival Drug Trends Analysis

Thank you for your interest in contributing to this project! This document provides guidelines for contributing.

## Ways to Contribute

### 1. Code Contributions

- **New Analysis Methods**: Add statistical tests, forecasting models, or additional metrics
- **Visualization Enhancements**: Create interactive plots, dashboards, or new chart types
- **Performance Improvements**: Optimize data processing or visualization generation
- **Bug Fixes**: Fix any issues you encounter

### 2. Documentation

- Improve README clarity
- Add more usage examples
- Create tutorials or guides
- Document methodology in more detail

### 3. Data Integration

- Adapt code to work with real drug checking datasets
- Add data validation and cleaning functions
- Create data import utilities

### 4. Research Applications

- Apply methodology to real-world datasets (with appropriate approvals)
- Extend analysis to other regions or contexts
- Add comparative studies with other harm reduction approaches

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/fixed-vs-festival-drug-trends-au.git
   cd fixed-vs-festival-drug-trends-au
   ```

3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run tests to verify setup:
   ```bash
   python test_pipeline.py
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Comment complex logic
- Keep functions focused and modular

## Testing

- Add tests for new functionality
- Ensure all existing tests pass
- Test with different data scenarios
- Verify visualizations render correctly

## Pull Request Process

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear, atomic commits

3. Test your changes thoroughly:
   ```bash
   python test_pipeline.py
   python main.py
   ```

4. Update documentation as needed

5. Submit a pull request with:
   - Clear description of changes
   - Motivation/use case
   - Screenshots for visual changes
   - Reference to any related issues

## Data Ethics

When working with real drug checking data:

- **Privacy**: Never commit personally identifiable information
- **Anonymization**: Ensure all data is properly de-identified
- **Consent**: Only use data with appropriate permissions
- **Ethics Approval**: Obtain necessary ethics approvals for research
- **Data Sharing**: Respect data sharing agreements

## Research Ethics

This tool is designed for harm reduction research. When publishing results:

- Cite the repository appropriately
- Acknowledge data sources
- Follow academic integrity standards
- Consider public health implications
- Engage with affected communities

## Feature Requests

Have an idea? Open an issue with:
- Clear description of the feature
- Use case or motivation
- Example of how it would work
- Any relevant references

## Bug Reports

Found a bug? Open an issue with:
- Description of the problem
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version)
- Error messages or logs

## Questions?

For questions about:
- **Using the code**: Open a GitHub issue
- **Methodology**: Check EXAMPLES.md or open an issue
- **Research collaboration**: Contact through GitHub

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Support harm reduction principles
- Prioritize public health over other considerations
- Recognize diverse perspectives

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.

## Attribution

Contributors will be acknowledged in the project. Significant contributions may warrant co-authorship on any resulting publications.

---

Thank you for contributing to harm reduction research and drug checking service evaluation!
