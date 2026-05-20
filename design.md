Design principles:
New features should not break existing features
Simplify existing code is better than adding code 
Use a loop if dealing with multiple items to reduce line of code
Extract shared logic across features or files into helper, if reasonable
Adding to specific ui components file is better than adding to ccg_ui.py
Adding to manager classes in ccg_ui.py is better than adding to CCGReviewUI main class
Consider adding attributes to existing classes over creating extra methods
Consider creating extra classes, if strictly needed, over adding extra methods
UI logic not specific to this project should live in ui utils.py
Consider extracting UI components into reusable classes
Maintain extensibility and modularity if you were to migrate interface to another UI package
Computation code should live in analyses/ms_connectivity.py
Concise comments and doc strings
Methods that directly tie to a UI component should be named after that UI component to speed up searching