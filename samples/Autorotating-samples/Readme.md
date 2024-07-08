### AutoRotate Plugin README

The AutoRotate plugin is designed to automatically rotate an image based on a reference image using template matching techniques. This README provides instructions on how to use the AutoRotate plugin with the provided JSON schema.

#### Configuration Options

- **name**: Specifies the name of the plugin. Must be set to `"AutoRotate"` for the configuration to be recognized as applying to AutoRotate.

- **options**: Object containing configuration options for AutoRotate.

- **referenceImage**: Relative path to the reference image used for template matching.

- **markerDimensions**: Dimensions of the reference image. This should be specified as an object containing two positive numbers (see `$def/two_positive_numbers` in your schema).

- **threshold**: Object defining the threshold parameters for the match score.

- **value**: Numeric threshold value. If the match score falls below this value, an error or warning may be triggered based on the `passthrough` setting.

- **passthrough**: Boolean indicating whether to pass through the image without rotation if the score is below the threshold (`true`) or to throw an warning (`false`) for an error

#### Example Configuration

```json
{
  "name": "AutoRotate",

  "options": {
    "referenceImage": "./path/to/reference_image.png",

    "markerDimensions": [100,150],

    "threshold": {
      "value": 100,

      "passthrough": true
    }
  }
}
```

#### Notes

- Ensure that the `name` field is set to `"AutoRotate"` to activate the plugin.

- Provide the correct relative path (`referenceImage`) to your reference image.

- Adjust `markerDimensions` to match the dimensions of your reference image accurately.

- Configure `threshold` according to your application's requirements for acceptable match scores.

This README should guide you through configuring and using the AutoRotate plugin effectively with your JSON schema-based configuration files. Adjust parameters as necessary to achieve the desired automatic image rotation based on template matching with a reference image.
