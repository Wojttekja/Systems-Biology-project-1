# Structure of Action Plan

---

## Short Explanation

**Explains How To Add & Manage Action Plan Items**

---

## Mid Explanation

Every Item should contain 5 Parts:

- Title (Must Have)

- Short Explanation (Must Have)

- Mid Explanation (Must Have)

- Long Explanation (Must Have - may be [TODO/NoNeed])

- Additional Description (Must Have - may be Empty)

Every Item should be marked by Atleast one of the Internal Labels.

---

## Long Explanation

### 1. Title 

- Should preceded by `#` symbol. 

- Should be ended with `---`.

- For Example:

```
# Title

---
```

### 2. Short Explanation

- Short Explanation should be announced by `## Short Explanation` text.

- Should contain max 2 sentences, shortly explaining what it is.

- Should be ended with `---`.

- For Example:

```
## Short Explanation

This item is doing that and is an example.

---
```

### 3. Mid Explanation

- Mid Explanation should be announced by `## Mid Explanation` text.

- Should not be long, but reading this must give understanding how this item shoould work and what is it purpouse.

- Should be ended with `---`.

- For Example:

```
## Mid Explanation

This item is doing that and that.

Used in this and this.

Is an example and it is located here and here.

It importance is that and that.

---
```

### 4. Long Explanation

- Long Explanation may not be needed if item is simple or may be written in future phases od item development. In that case should be marked as 
   - [TODO] / [NoNeed] / [TODO or NoNeed] according to **TODO management** section.

- May be as long as it needs to be. But Unnecesary for Item functionality Information should be in *5. Additional Description*. 

- Should Contain Implementation Details (meaning *description in human terms*)

- May contain *short* Code Fragments and Math equations. But those should be located in full description (meaning in *Extended Readme* Folder)

- Should be ended with `---`.

- For Example:

```
## Long Explanation

This item is doing that and that.

Used in this and this.

Is an example and it is located here and here.

It importance is that and that.

Implementation is that and that.

[...]

Math is that and that.

[...]

---
```

### 5. Additional Descriptions

- It contains mostly explanations why given *label* is assigned or its just additional information given by Author.

- Explanation must contain `[explanation short] Desc` section + other sections if Author assumes that they are needed.

- Explanations should be ended with `---`.

- Should be present in every item, and should have explanation for every *label*. But those explanations may be marked as:
   - [TODO] / [NoNeed] / [TODO or NoNeed] according to **TODO management** section.

- Should be ended with `---`.

- For Example:

```
## Additional Descriptions

---

### Additional Desc [1]

#### Design Desc:

[...]

---

### Additional Desc [2]

#### Design Desc:

[TODO or NoNeed]

---
```


### TODO Management Section:
   - [TODO] - When it is planned to add TODO.
   - [NoNeed] - When there is no need for any Explanation.
   - [TODO or NoNeed] - When TODO presence is Unknown

### Internal Labels:
   - They start with `[R]` which stands for `Repository`.
   - Are used for fast distinction and classification of Project items.
   - Every Item should have assigned atleast *one* label.

### EasyTemplate:
   - There is a posibility to use Easy Template. 
   - However it should be refactored according to the Intructions in this document.

---

# Additional Descriptions

[TODO or NoNeed]

---
