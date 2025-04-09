import nopecha
nopecha.api_key = 'YOUR_API_KEY'

# Call the Recognition API
clicks = nopecha.Recognition.solve(
    type='recaptcha',
    task='Select all squares with vehicles.',
    image_urls=['https://nopecha.com/image/demo/recaptcha/4x4.png'],
    grid='4x4'
)

# Print the grids to click
print(clicks)