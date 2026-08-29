# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in VIDEOContext, please do not disclose it publicly before it has been reviewed.

Instead, report the vulnerability privately to the project maintainer.

Please include as much information as possible, including:

* A description of the vulnerability
* The affected component or functionality
* Steps to reproduce the issue
* Potential impact
* Proof of concept, if available
* Suggested mitigation or fix, if you have one

Please avoid including sensitive information that is not necessary to understand or reproduce the issue.

## What to Expect

Security reports will be reviewed when possible.

The project maintainer may request additional information to better understand and reproduce the issue.

If the vulnerability is confirmed, an appropriate fix will be developed and released based on the severity and impact of the issue.

Please understand that response and resolution times may vary, particularly as VIDEOContext is an open-source project.

## Supported Versions

Security fixes are generally applied to the actively maintained version of VIDEOContext.

| Version                    | Supported          |
| -------------------------- | ------------------ |
| Latest development version | :white_check_mark: |
| Older versions             | :x:                |

If you are using an older version of VIDEOContext, updating to the latest available version is recommended before reporting an issue unless the vulnerability specifically prevents you from doing so.

## Scope

Potential security issues may include, but are not limited to:

* Arbitrary code execution
* Unsafe file handling
* Path traversal
* Malicious video or media file processing
* Denial of service
* Dependency vulnerabilities
* API authentication or authorization issues
* Exposure of sensitive information
* Unsafe handling of uploaded files
* Injection vulnerabilities
* Vulnerabilities affecting the MCP server or tools

## Out of Scope

The following are generally not considered security vulnerabilities:

* Issues requiring unrealistic attack conditions
* Theoretical issues without a practical security impact
* Missing security features that do not expose an existing vulnerability
* Issues in third-party services or dependencies that are outside the control of VIDEOContext, unless VIDEOContext directly introduces or enables the vulnerability

## Handling Uploaded Video Files

VIDEOContext may process user-provided video and media files.

Media files should be treated as untrusted input.

Contributors working on video processing functionality should consider security implications related to:

* File size
* File type validation
* Malformed media files
* Resource exhaustion
* Unexpected metadata
* Path handling
* External media processing tools

Avoid assuming that uploaded files are safe or well-formed.

## Dependencies

VIDEOContext may depend on third-party libraries and external tools for media processing, AI functionality, or other features.

When reporting a dependency-related vulnerability, please include:

* The affected dependency
* The affected version
* How VIDEOContext is impacted
* Whether the issue can be triggered through normal VIDEOContext functionality

## Responsible Disclosure

Please allow reasonable time for a reported vulnerability to be reviewed and addressed before publicly disclosing technical details.

If the vulnerability is confirmed and fixed, public disclosure may occur after users have had an opportunity to update.

## Security Best Practices for Users

When running VIDEOContext:

* Keep dependencies updated
* Use trusted and supported versions where possible
* Treat uploaded media files as untrusted
* Avoid exposing development services directly to the public internet without appropriate security controls
* Protect API keys and credentials using environment variables or secure secret-management systems
* Review access controls before deploying VIDEOContext in production

## Thank You

Responsible vulnerability reports help improve the security of VIDEOContext and protect its users.

Thank you for helping make VIDEOContext more secure.
