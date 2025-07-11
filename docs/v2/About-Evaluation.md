# The evaluation.json file

This documentation provides a clear overview of the `evaluation.json` structure and its components, enabling efficient customization and understanding of the evaluation process within OMRChecker. Adjustments and expansions to meet specific evaluation needs can be made based on these defined parameters and configurations.

## Meaning of Parameters Used in the `evaluation.json` File

To understand the structure and usage of the `evaluation.json` file for custom evaluation processes, let's delve into its parameters and configurations. Below is an annotated explanation of each section:

**Note: Add evaluation.json at the same folder level as your template.json**

### List of capabilities

- multiple marking schemes including negative/fractional
- colored outputs with printing score,
- set-mapping of answer keys

<p align="center">
	<a href="https://github.com/Udayraj123/OMRChecker/wiki/%5Bv2%5D-About-Evaluation">
		<img alt="colored_output" width="550" src="./images/colored_output.jpg">
	</a>
</p>

### Source Types and Answer Key Options

- **source_type**: Specifies the type of source data being evaluated, in this case, it's "local". We also support `csv` and `image_and_csv` sources. Please refer to the [samples](https://github.com/Udayraj123/OMRChecker/tree/master/samples/3-answer-key/) folder for more examples.
- **questions_in_order**: Defines the questions in a serial order in the evaluation.
- **answers_in_order**: Defines the answers expected in a serial order for each question in the evaluation.

##### Types of answer keys supported:

1. Standard answer type: allows single correct answers. They can have multiple characters(multi-marked) as well.
   Useful for any standard response e.g. 'A', '01', '99', 'AB', etc
2. Multiple correct answer type: covers multiple correct answers
   Useful for ambiguous/bonus questions e.g. `['A', 'B'], ['1', '01'], ['A', 'B', 'AB']`, etc
3. Multiple correct weighted answer: covers multiple answers with custom scores
   Useful for partial marking e.g. `[['A', 2], ['B', 0.5], ['AB', 2.5]], [['1', 0.5], ['01', 1]]`, etc

```js
{
  "source_type": "local",
  "options": {
    // The field names mentioned in 'questions_in_order' are picked up from the OMR response.
    "questions_in_order": [
      "q1..10", // A field string indicating questions 1 to 11 (for default section)
      "s2_q1..5", // Another field string indicating questions 1 to 5 from another section
    ],
    // Various answer formats for each question
    "answers_in_order": [
        "A", // q1: Single correct answer 'A'
        "B", // q2: Single correct answer 'B'
        "AB", // q3: Multicharacter answer 'AB' (both A & B should be marked)
        ["A", "B"], // q4: Multiple correct answers (either A or B is correct but not both)
        ["A", "B", "AB"], // q5: Multiple correct answers (either A or B or both AB is correct)
        ["A", "B", "C", "D"], // q6: Effective bonus marks using custom score (without using bonus section)
        [["A", 2], ["D", 1], ["AD", 3]], // q7: Multiple answers with custom score (Marks for AD = 2 + 1 = 3)
        [["C", 10], ["D", -1]], // q8: Multiple answers with custom negative score
        "D", // q9: Original answer 'D' (But score overridden by bonus marking scheme below)
        "C", // q10: Original answer 'C' (But score overridden by bonus marking scheme below)
    ]
    "marking_schemes":{
        // Default marking scheme (fallback)
        "DEFAULT": {
            "correct": "3", // Default marking for correct answers
            "incorrect": "0", // Default marking for incorrect answers
            "unmarked": "0" // Default marking for unmarked answers
        },
        // Custom section with a different marking scheme
        "SECTION_2":{
            // List of questions to apply the marking scheme on
            "questions":["s2_q1..5"],
            // Custom marking on the section questions
            "marking": {
                "correct": "4",
                "incorrect": "-1",
                "unmarked": "0"
            }
        },
        // Another section mentioning bonus questions(on attempt)
        "BONUS_MARKS_ON_ATTEMPT": {
            "questions": [
                "q9"
            ],
            // Custom marking on the section questions
            "marking": {
                "correct": "3",
                "incorrect": "3",
                "unmarked": "0"
            }
        },
        // Another section mentioning bonus questions(with/without attempt)
        "BONUS_MARKS_FOR_ALL": {
            "questions": [
                "q10"
            ],
            "marking": {
                "correct": "3",
                "incorrect": "3",
                "unmarked": "3"
            }
        },
    }
  },
```

### Symbols and Color Notations 
When `draw_question_verdicts` is enabled, the output images will contain the answer key and corresponding question verdicts. Both the gray and colored images follow a set of notations to show the different types of answers supported. The diagram below(initial draft) explains the possible answer key cases covered in a single picture:
![evaluation-outputs-export](https://github.com/Udayraj123/OMRChecker/assets/16881051/844895f4-c3ce-47dc-9688-60cd9bc6a3e3)


Note: As of now the output notations for keys using `customLabels` are yet to be supported and will not be shown visually, but used in the score calculation directly.


### Outputs Configuration

- **should_explain_scoring**: Indicates whether to print a table explaining question-wise verdicts.
- **draw_score**: Configuration for drawing the final score, including its position and size.
- **draw_answers_summary**: Configuration for drawing the answers summary, including its position and size.
- **draw_question_verdicts**: Configuration for drawing question verdicts, specifying colors and symbols for different verdict types.
- **draw_detected_bubble_texts**: Configuration for drawing detected bubble texts, which is disabled in this example.


```js
{
  "outputs_configuration": {
    "should_explain_scoring": true, // Whether to explain question-wise verdicts
    "draw_score": {
      "enabled": true, // Enable drawing the score
      "position": [600, 650], // Position of the score box
      "size": 1.5 // Font size of the score box
    },
    "draw_answers_summary": {
      "enabled": true, // Enable drawing answers summary
      "position": [300, 550], // Position of the answers summary box
      "size": 1.0 // Font size of the answers summary box
    },
    "draw_question_verdicts": {
      "enabled": true, // Enable drawing question verdicts

      // Colors for different verdicts
      "verdict_colors": {
        "correct": "lightgreen", // Color for correct answers
        "neutral": "#000000", // Color for neutral verdicts (delta == 0)
        "incorrect": "#ff0000", // Color for incorrect answers
        "bonus": "#00DDDD" // Color for bonus questions
      },

      // Colors for different verdict symbols
      "verdict_symbol_colors": {
        "positive": "#000000", // Color for '+' symbol (delta > 0)
        "neutral": "#000000", // Color for 'o' symbol (delta == 0)
        "negative": "#000000", // Color for '-' symbol (delta < 0)
        "bonus": "#000000" // Color for '*' symbol (bonus question)
      },

      // Configuration for drawing answer groups
      "draw_answer_groups": {
        "enabled": true // Enable drawing answer groups
      }
    },
    "draw_detected_bubble_texts": {
      "enabled": false // Disable drawing detected bubble texts
    }
  },
}
```

### Example Usage

To load and utilize this schema, navigate to the project root and execute the following command:

```bash
python3 main.py -i samples/3-answer-key/bonus-marking-grouping
```

For further details and examples, refer to the [samples](https://github.com/Udayraj123/OMRChecker/tree/master/samples/3-answer-key/) directory.
