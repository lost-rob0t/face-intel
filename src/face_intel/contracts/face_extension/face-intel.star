(spec-library "org.starintel/face-intel@1"
  (:version "0.1.0")

  (import "org.starintel/core@1"
    :version "0.10.1"
    :digest "sha256:e09205e71fcd0f2bf87ff7d104c5ccff298d0054764def81e00f81a6bd8639fc")

  ;; A supplied observation and an identity claim are separate documents.
  ;; Persisting an annotation does not establish a person's identity.
  (scalar pixel-offset (:base integer :minimum 0))
  (scalar pixel-length (:base integer :minimum 1))
  (scalar annotation-basis (:base string :pattern "\\S"))
  (enum candidate-only (candidate))

  (document face-observation
    (:extends org.starintel/core@1/document :persistence persistent)
    (picture reference :required)
    (x pixel-offset :required)
    (y pixel-offset :required)
    (width pixel-length :required)
    (height pixel-length :required)
    (annotationBasis annotation-basis :required))

  (document candidate-person
    (:extends org.starintel/core@1/person :persistence persistent)
    (candidateStatus candidate-only :required)
    (annotationBasis annotation-basis :required)
    (evidenceReferences (list reference) :optional))

  (document face-person-candidate
    (:extends org.starintel/core@1/relation :persistence persistent)
    (candidateStatus candidate-only :required)
    (annotationBasis annotation-basis :required)
    (evidenceReferences (list reference) :optional))

  (predicate candidate-person-for-face
    (:source face-observation :destination org.starintel/core@1/person))

  ;; These describe annotation storage, not matching or identity inference.
  ;; HTTP, RabbitMQ, and StarRouter adapters can carry the same documents.
  (message store-face-observation
    (:fields ((document reference :required))))
  (message store-candidate-person
    (:fields ((document reference :required))))
  (message annotate-face-person
    (:fields ((face reference :required)
              (person reference :required)
              (annotationBasis annotation-basis :required))))
  (message get-face-observation
    (:fields ((face reference :required)))))
